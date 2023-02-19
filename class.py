import os
import pandas as pd
import requests

class Group22:
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename
        
    def download_data(self):
        # Download the data file if it is not already in the directory
        if not os.path.exists(os.path.join('downloads', self.filename)): 
            response = requests.get(self.url)
            if response.status_code == 200:
                with open(os.path.join('downloads', self.filename), 'wb') as f:
                    f.write(response.content)
                print(f'{self.filename} has been downloaded')
            else:
                print(f'Error downloading {self.filename}: {response.status_code}')
        else: 
            print(f'{self.filename} already exists')
        
        # Read the data file into a DataFrame
        data_path = os.path.join('downloads', self.filename)
        df = pd.read_csv(data_path, on_bad_lines='skip')
        
        return df