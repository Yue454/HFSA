import torch
from torch import Tensor
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

import pywt
from torch.autograd import Function
import torch.nn.functional as F

__all__ = ['de_resnet18', 'de_resnet34', 'de_resnet50',
           'de_wide_resnet50_2', 'de_wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
              dilation: int = 1) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False,
                              dilation=dilation)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, upsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if stride == 2:
            self.conv1 = deconv2x2(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None: identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, upsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.upsample is not None: identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi, dec_lo = torch.tensor(w.dec_hi[::-1], dtype=type), torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1), dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1), dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)],
                              dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
    rec_hi, rec_lo = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0]), torch.tensor(w.rec_lo[::-1],
                                                                                           dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1), rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1), rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)],
                              dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    return x.reshape(b, c, 4, h // 2, w // 2)


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)


def wavelet_transform_init(filters):
    class WaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            return wavelet_transform(input, filters)

        @staticmethod
        def backward(ctx, grad_output):
            return inverse_wavelet_transform(grad_output, filters), None

    return WaveletTransform().apply


def inverse_wavelet_transform_init(filters):
    class InverseWaveletTransform(Function):
        @staticmethod
        def forward(ctx, input):
            return inverse_wavelet_transform(input, filters)

        @staticmethod
        def backward(ctx, grad_output):
            return wavelet_transform(grad_output, filters), None

    return InverseWaveletTransform().apply


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)


class MultiFrequencyResponseModule(nn.Module):
    def __init__(self, in_channels, kernel_size=5, wt_levels=1, wt_type='db1'):
        super(MultiFrequencyResponseModule, self).__init__()
        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter, self.iwt_filter = nn.Parameter(self.wt_filter, False), nn.Parameter(self.iwt_filter, False)
        self.wt_function, self.iwt_function = wavelet_transform_init(self.wt_filter), inverse_wavelet_transform_init(
            self.iwt_filter)
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels, bias=True)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList([nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same',
                                                      groups=in_channels * 4, bias=False) for _ in range(wt_levels)])
        self.wavelet_scale = nn.ModuleList([_ScaleModule([1, in_channels * 4, 1, 1], 0.1) for _ in range(wt_levels)])

    def forward(self, x):
        x_ll_in, x_h_in, shapes = [], [], []
        curr_x_ll = x
        for i in range(self.wt_levels):
            shapes.append(curr_x_ll.shape)
            if (curr_x_ll.shape[2] % 2 > 0) or (curr_x_ll.shape[3] % 2 > 0):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_x_ll.shape[3] % 2, 0, curr_x_ll.shape[2] % 2))
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], -1, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)).reshape(shape_x)
            x_ll_in.append(curr_x_tag[:, :, 0, :, :])
            x_h_in.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x = torch.cat([(x_ll_in.pop() + next_x_ll).unsqueeze(2), x_h_in.pop()], dim=2)
            next_x_ll = self.iwt_function(curr_x)
            curr_shape = shapes.pop()
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        x = self.base_scale(self.base_conv(x)) + next_x_ll
        return x


class FSFD(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int],
                 norm_layer: Optional[Callable[..., nn.Module]] = None, **kwargs) -> None:
        super(FSFD, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = kwargs.get('groups', 1)
        self.base_width = kwargs.get('width_per_group', 64)


        self.lfe_block1 = self._make_lfe_layer(block, 256, layers[0], stride=2)
        self.lfe_block2 = self._make_lfe_layer(block, 128, layers[1], stride=2)
        self.lfe_block3 = self._make_lfe_layer(block, 64, layers[2], stride=2)

        self.mrfm_block1 = MultiFrequencyResponseModule(in_channels=2048)
        self.mrfm_block2 = MultiFrequencyResponseModule(in_channels=1024)
        self.mrfm_block3 = MultiFrequencyResponseModule(in_channels=512)

        self.upsample_mrfm1 = deconv2x2(2048, 1024, 2)
        self.upsample_mrfm2 = deconv2x2(1024, 512, 2)
        self.upsample_mrfm3 = deconv2x2(512, 256, 2)

        self.fusion_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        self.fusion_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.fusion_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def _make_lfe_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1,
                        dilate: bool = False) -> nn.Sequential:
        norm_layer, upsample, previous_dilation = self._norm_layer, None, self.dilation
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(deconv2x2(self.inplanes, planes * block.expansion, stride),
                                     norm_layer(planes * block.expansion))
        layers = [
            block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> List[Tensor]:


        mrfm_stream1 = self.upsample_mrfm1(self.mrfm_block1(x))
        lfe_stream1 = self.lfe_block1(x)
        feature_a = self.fusion_conv1(torch.cat([mrfm_stream1, lfe_stream1], 1))

        mrfm_stream2 = self.upsample_mrfm2(self.mrfm_block2(feature_a))
        lfe_stream2 = self.lfe_block2(feature_a)
        feature_b = self.fusion_conv2(torch.cat([mrfm_stream2, lfe_stream2], 1))

        mrfm_stream3 = self.upsample_mrfm3(self.mrfm_block3(feature_b))
        lfe_stream3 = self.lfe_block3(feature_b)
        feature_c = self.fusion_conv3(torch.cat([mrfm_stream3, lfe_stream3], 1))


        # feature_a = self.lfe_block1(x)
        # feature_b = self.lfe_block2(feature_a)
        # feature_c = self.lfe_block3(feature_b)


        # feature_a = self.upsample_mrfm1(self.mrfm_block1(x))
        # feature_b = self.upsample_mrfm2(self.mrfm_block2(feature_a))
        # feature_c = self.upsample_mrfm3(self.mrfm_block3(feature_b))


        return [feature_c, feature_b, feature_a]

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)


def _decoder(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool,
             **kwargs: Any) -> FSFD:
    model = FSFD(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # Custom logic to load weights if necessary, but typically decoders are not pretrained
        # model.load_state_dict(state_dict, strict=False)
    return model


def de_resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> FSFD:
    return _decoder('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def de_resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> FSFD:
    return _decoder('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def de_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> FSFD:
    return _decoder('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def de_wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> FSFD:
    kwargs['width_per_group'] = 64 * 2
    return _decoder('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def de_wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> FSFD:
    kwargs['width_per_group'] = 64 * 2
    return _decoder('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)