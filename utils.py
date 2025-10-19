import requests

def download_images(image_urls, prefix='image_', max_images=5):
    """
    Downloads images from a list of URLs.
    Saves them in the current folder as image_0.jpg, image_1.jpg, etc.
    Downloads up to max_images images for demo/testing.
    
    Args:
        image_urls (list): List of image URLs (str).
        prefix (str): Filename prefix for images.
        max_images (int): Number of images to download (prevents overload).
    """
    for i, url in enumerate(image_urls[:max_images]):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(f"{prefix}{i}.jpg", "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {prefix}{i}.jpg")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
