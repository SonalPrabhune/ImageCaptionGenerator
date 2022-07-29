import ImageCaptionGenerator
from app import app

if __name__ == '__main__':
    icgen = ImageCaptionGenerator.ImageCaptionGenerator
    icgen.generateCaption(icgen)
    #app.run(port=12345, debug=True)


