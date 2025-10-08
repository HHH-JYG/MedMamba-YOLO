from .common_utils_mbyolo import *
from .atten import *
import torch.nn.functional as F

__all__ = ['FusionMamba']


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96, # 输出特征维度
            dim=150, #输入特征维度
            d_state=16, # 状态空间模型的维度，控制模型的复杂度。
            ssm_ratio=1.0, #特征扩展倍数，决定扩展后的维度。
            ssm_rank_ratio=1.0, #扩展维度低秩近似的比率，用于控制计算复杂度。
            dt_rank="auto", #时间常数秩，默认为 auto，动态确定。
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv # 卷积核大小
            conv_bias=True, #是否在卷积层中使用偏置。
            # ======================
            dropout=0.0,
            bias=False, #是否为线性投影层添加偏置。
            # ======================
            forward_type="v2", #前向传播方式的选择（如 v2）。
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        d_expand = int(ssm_ratio * d_model)   # 通过 ssm_ratio 扩展输入特征维度
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand  # 扩展后的内部低秩特征维度

        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank   # 根据输入特征自动计算时间常数秩，或使用指定值。
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 根据输入特征自动计算状态空间维度，或使用指定值。
        self.d_conv = d_conv   # 卷积核大小
        self.K = 4

        # tags for forward_type ==============通过检测 forward_type 的后缀，设置前向传播的不同标志。
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag # 检查字符串 value 是否以 tag 结尾
            if ret:
                value = value[:-len(tag)] # 如果匹配，移除后缀 tag
            return ret, value # 返回匹配结果和移除后缀后的字符串（若无匹配，原样返回）

        self.disable_force32, forward_type = checkpostfix("no32", forward_type) #表示模型在训练中不强制使用 FP32（浮点数精度 32 位）。
        self.disable_z, forward_type = checkpostfix("noz", forward_type) #表示禁用 Z 分支
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type) #表示不对 Z 分支使用激活函数（如 ReLU、GELU 等）
        self.out_norm = nn.LayerNorm(d_inner)
        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )

        """
        根据 forward_type 的值，选择相应的前向传播函数，赋值给 self.forward_core。
        如果 forward_type 无法匹配，则默认使用 v2 类型。
        当 forward_type="v2" 时，直接匹配字典中的 v2 条目。
        此时self.forward_core = partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore)。
        参数介绍
        force_fp32=None: 强制使用 FP32 的标志（默认为 None）
        SelectiveScan=SelectiveScanCore: 前向传播时使用的扫描核心函数。
        """
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))


        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        # d_proj = d_expand 

        # 线性投影层，将输入特征扩展为
        self.in_proj = nn.Conv2d(dim, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(  #启用深度卷积
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand, # 分组数等于通道数，启用 Depthwise Convolution
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False #如果内部维度 d_inner 小于扩展后的维度 d_expand，启用低秩近似。
        if d_inner < d_expand: #相等
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs) #一个 1×1 的卷积层，用于将 d_expand 维度降到 d_inner。
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs) #一个线性层，将 d_inner 恢复到 d_expand。

        # x proj ============================
        """
        self.K:表示多投影头的数量（通常用于提高表达能力）。
        weight 的形状是 (out_features, in_features)
        """
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        """
        将 self.K 个投影层的权重矩阵堆叠成一个 3D 张量。
        形状为 (K, (dt_rank + d_state * 2), d_inner)。
        删除原始的 x_proj 列表，仅保留堆叠的权重参数。
        """
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj ====使用 1x1 卷积实现从扩展维度 (d_expand) 到原始输入特征维度 (d_model) 的线性映射。
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner))) #初始化跳跃参数
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.norm_1 = norm_layer(d_expand)
        self.SpatialAttention = SpatialAttention(d_expand)
       

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant": #使用 dt_init_std 常数初始化权重。
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random": #在 [-dt_init_std, dt_init_std] 的范围内均匀随机初始化权重。
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj  #返回初始化完成的线性层 dt_proj

    """
    初始化矩阵 𝐴的对数形式 (log(A))，这是一个可训练的参数，通常用于时间相关模型中对状态矩阵的初始化。
    """
    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    """
    是模型的核心前向传播函数，结合了多个复杂的操作，例如投影、选择性扫描（Selective Scan），以及低秩分解等方法。
    channel_first：指定输入通道是否在第一维度（默认 False）。
    SelectiveScan：自定义选择性扫描函数，默认为 SelectiveScanCore。
    cross_selective_scan：自定义交叉选择性扫描函数，默认为 cross_selective_scan。
    """
    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        #如果 force_fp32 未显式指定，则在训练模式下并且 disable_force32=False 时，默认启用 FP32。
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32

        #下面两个不执行
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        """
        self.x_proj_weight：权重矩阵，用于线性变换。
        self.dt_projs_weight & self.dt_projs_bias：时间相关参数，用于动态调整特征表示.
        self.A_logs：对数形式的矩阵，用于状态动态调整。
        self.Ds：状态向量。
        SelectiveScan：自定义扫描方法。
        """
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"), #如果 self 没有 out_norm_shape 属性，就会使用默认值 "v0"。
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        x = self.SpatialAttention(x)
        
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        y = self.norm_1(y)
        out = self.dropout(self.out_proj(y))
        return out

class FusionMamba(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            concat_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=1.0,
            ssm_rank_ratio=1.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0  #True
        self.mlp_branch = mlp_ratio > 0  #True
        self.use_checkpoint = use_checkpoint  #False
        self.post_norm = post_norm  #False

        if self.ssm_branch:
            self.norm = norm_layer(concat_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                dim=concat_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                norm_layer=norm_layer,
            )

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        length = len(x)
        if length == 3:
            target_size = x[1].shape[2:]
            x[0] = F.interpolate(x[0], size=target_size, mode='nearest') 
            x[2] = F.interpolate(x[2], size=target_size, mode='nearest') 
            input = torch.cat(x, 1)
            input = self.drop_path(self.op(self.norm(input)))
        elif length ==2:
            target_size = x[1].shape[2:]
            x[0] = F.interpolate(x[0], size=target_size, mode='nearest') 
            input = torch.cat(x, 1)
            input = self.drop_path(self.op(self.norm(input)))
        return x[1]+input






