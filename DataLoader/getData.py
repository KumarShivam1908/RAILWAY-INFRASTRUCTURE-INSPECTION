from roboflow import Roboflow

class DataLoader:
    def __init__(self, api_key="58BX6EZLXZvWhy5sZjLn"):
        """Initialize DataLoader with API key."""
        self.api_key = api_key
        self.rf = None
        self.project = None
        self.version = None
        self.dataset = None

    def setup_roboflow(self):
        """Setup Roboflow connection."""
        try:
            self.rf = Roboflow(api_key=self.api_key)
            self.project = self.rf.workspace("bridge-damage-detection-using-yolov5").project("bridge-damage-detection")
            return True
        except Exception as e:
            print(f"Error setting up Roboflow: {e}")
            return False

    def download_dataset(self, version_num=2, format="yolov11"):
        """Download dataset with specified version and format."""
        try:
            if not self.project:
                if not self.setup_roboflow():
                    return False
            
            self.version = self.project.version(version_num)
            self.dataset = self.version.download(format)
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False


if __name__ == "__main__":
    loader = DataLoader()
    loader.download_dataset()



                