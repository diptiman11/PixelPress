PixelPress: Image Compression using K-Means Clustering
PixelPress is a machine learning project implementing lossy image compression using the K-Means Clustering algorithm. It compresses .png and .jpg images, reducing file size while maintaining acceptable visual quality.

Features:
Compress .png and .jpg images using K-Means Clustering.
Decompress images back to a visually similar form.
## To Use
Usable for a `.png` and `.jpg` image.

### For Compression
Follow the following steps:
1. Select your image to be compressed having valid file format.
2. Get your image file path by right clicking->Properties->Location. Copy it.
3. Run the cpp application executable file.
4. Enter the `compress` command and paste the copied path (Wait 10 seconds).

You should now have two new files alongside the image namely `filename_codebook.npy` and `filename_compressed.png`.
These files are the compressed form of your image.

### For Decompression
Follow the following steps:
1. Get the filepath of your `filename_compressed.png` file as previously mentioned.
2. Run the cpp executable file.
3. Enter the `decompress` command and paste the copied path.

You should now have a new file in the directry as `filename_decompressed.png`.
This is the lossy decompressed version of your original image.#   P i x e l P r e s s 
 
 
