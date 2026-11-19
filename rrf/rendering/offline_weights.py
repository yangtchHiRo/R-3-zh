"""Allow cache reads by the original LPIPS implementation, forbid downloads."""


def block_downloads(torch):
    import urllib.request

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "LPIPS/AlexNet weights are not available in the current cache. "
            "Use the same pgnd environment, HOME and TORCH_HOME as your successful image evaluation. "
            "No download was performed; do not use random LPIPS weights."
        )

    torch.hub.download_url_to_file = blocked
    urllib.request.urlretrieve = blocked
    urllib.request.urlopen = blocked
