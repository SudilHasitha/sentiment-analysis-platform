import subprocess

def install_ffmpeg():
    print("Installing ffmpeg...")
    subprocess.run(['python3', '-m', 'pip', 'install', '--upgrade','pip'], check=True)
    subprocess.run(['python3', '-m', 'pip', 'install', '--upgrade','setuptools'], check=True)
    try:
        subprocess.run(['python3', '-m', 'pip', 'install', 'ffmpeg'], check=True)
        print("FFmpeg installed successfully")
    except Exception as e:
        print(f"Error installing ffmpeg: {e} via pip")

    
    # install ffmpg via static build
    try:
        subprocess.run(['wget', 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz',
         '-O', '/tmp/ffmpeg-release-amd64-static.tar.xz'], check=True)
        subprocess.run(['tar', '-xvf', '/tmp/ffmpeg-release-amd64-static.tar.xz', '-C', '/tmp'], check=True)
        subprocess.run(['cp', '/tmp/ffmpeg-release-amd64-static/ffmpeg', '/usr/local/bin/ffmpeg'], check=True)
        # make ffmpeg executable
        subprocess.run(['chmod', '+x', '/usr/local/bin/ffmpeg'], check=True)
        # remove temporary files
        subprocess.run(['rm', '-rf', '/tmp/ffmpeg-release-amd64-static'], check=True)
        subprocess.run(['rm', '-rf', '/tmp/ffmpeg-release-amd64-static.tar.xz'], check=True)
        print("FFmpeg installed successfully via static build")
    except Exception as e:
        print(f"Error installing ffmpeg: {e} via static build")
        raise e 

    # verify ffmpeg is installed
    try:
        result = subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True)
        print(f"FFmpeg version: {result.stdout}")
    except Exception as e:
        print(f"Error verifying ffmpeg: {e}")
        raise e
