import os, tempfile, urllib.request
url = "http://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip"
out = os.path.expanduser("~/matterport3d/src/v1/tasks/mp3d_habitat.zip")
os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.isfile(out):
    print("already exists", out)
else:
    print("downloading", url)
    fh, tmp = tempfile.mkstemp(dir=os.path.dirname(out))
    os.close(fh)
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, out)
print("ok", out, os.path.getsize(out))
