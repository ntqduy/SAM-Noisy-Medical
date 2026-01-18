from datasets.image_mask_folder import ImageMaskFolderAdapter

_REG = {
    "ImageMaskFolderAdapter": ImageMaskFolderAdapter,
}

def build_dataset(cfg: dict):
    adapter = cfg["adapter"]
    if adapter not in _REG:
        raise ValueError(f"Unknown dataset adapter: {adapter}. Available: {list(_REG.keys())}")
    return _REG[adapter](cfg)
