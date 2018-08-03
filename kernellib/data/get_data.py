import requests
from scipy.io import loadmat


def generate_nystrom_data():

    data_url = 'https://github.com/mli/nystrom/raw/master/satimage.mat'
    r = requests.get(data_url, allow_redirects=True)
    open('satire.mat', 'wb').write(r.content)

    data = loadmat('satire.mat')['D'].toarray()

    return data
