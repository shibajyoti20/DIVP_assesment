from image_processors import *
import argparse

parser = argparse.ArgumentParser(description="Uml0dGlrIERhc2d1cHRhIEVDQjE5MDM1LCBQcmF0aW0gQ2hha3JhYm9ydHkgRUNCMTkwMjEK")

# Add an argument
parser.add_argument('-f', '--file', type=str, help="enter filepath for image processing",required=True)
parser.add_argument('-t', '--type', type=str, help="enter type of image processing",required=True)
parser.add_argument('-q', '--quality', type=int, help="enter desired image quality for image compression")
parser.add_argument('-k', '--clusters', type=int, help="enter desired image quality for image compression")


if __name__ == "__main__":


    # Parse the argument
    args = parser.parse_args()

    processing_type = args.type

    # Image Compression
    if(processing_type == "compress"):

        ImageCompressor.compress_normal(args.file, args.quality)
    

    # Image Transformation
    elif(processing_type == "dct"):

        ImageCompressor.dct_compress(args.file)
    elif(processing_type == "dft"):
        
        ImageCompressor.dft_compress(args.file)


    # Image Enhancement
    elif(processing_type == "histeq"):

        ImageEnhancer.histogram_eq(args.file)
    
    elif(processing_type == "clahe"):

        ImageEnhancer.clahe(args.file)
    

    # Image Segmentation
    elif(processing_type == "kmeans"):

        ImageSegmentor.k_means(args.file, args.clusters)
    
    elif(processing_type == "contour"):

        ImageSegmentor.contour_detection(args.file)

        