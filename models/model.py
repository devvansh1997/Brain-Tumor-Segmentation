from models.unet import UNet2D


def build_model(
    in_channels: int = 4,
    out_channels: int = 4,
    base_channels: int = 32,
):
    """
    Build the segmentation model.

    For BraTS slice-wise 2D segmentation:
    - in_channels = 4 MRI modalities
    - out_channels = 4 classes after remapping {0,1,2,4} -> {0,1,2,3}
    """
    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )
    return model