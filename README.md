# DeepFISH
Why Deep? Because of deep-learning.

Why FISH? Because of **F**luorescent **i**n-**s**itu **H**ybridization.

This repository is intended to share data and code for resolving some problems met in cytogenetics imaging such overlapping chromosomes.

## Problem description

   In cytogenetics, experiments typically starts from chromosomal preparations fixed on glass slides. Occasionally a chromosome can fall on another one, yielding overlapping chromosomes in the image. Before computers and images processing with photography, chromosomes were cut from a paper picture and then classified (at least two paper pictures were required when chromosomes are overlapping). Automatic segmentation methods were developped to overcome this problem, however, these methods rely on a geometric analysis of the chromosome contour and require some human intervention when partial overlap occurs.
   
   The [QFISH on metaphase](https://en.wikipedia.org/wiki/Q-FISH) was classified as a [low-throughput method for quantitative analysis of the lenght of the telomeres](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3409675/figure/F1/) by Vera and Blasco. One of the botleneck of the method is the resolution of the the overlapping chromosomes. 
Modern deep-learning techniques have the potential to provide a more reliable, fully-automated solution.


### Cytogenetics references
  * [A Geometric Approach To Fully Automatic Chromosome Segmentation](https://arxiv.org/abs/1112.4164)
  * [Automated Discrimination of Dicentric and Monocentric Chromosomes by Machine Learning-based Image Processing](http://biorxiv.org/content/biorxiv/early/2016/01/19/037309.full.pdf)
  * [An Efficient Segmentation Method for Overlapping Chromosome Images](http://research.ijcaonline.org/volume95/number1/pxc3894861.pdf)
  * [A Review of Cytogenetics and its Automation](http://www.scialert.net/qredirect.php?doi=jms.2007.1.18&linkid=pdf)

## Libraries required to run this notebook:

This notebook is run from [jupyter](http://jupyter.org/) with a python2 Kernel on a [Ubuntu 16.04 OS](https://www.ubuntu.com/desktop) inside a virtual environnement using the python packages available on the system. Several image processing libraries are used:

   * [mahotas](http://luispedro.org/software/mahotas/)
   * [opencv](http://opencv.org/)
   * [scipy](https://www.scipy.org/)
   * [Numpy](http://www.numpy.org/)
   * [scikit-image](http://scikit-image.org/)
    
### Notebooks

Up to now, there's only python notebooks is to produce a dataset large enough to train a supervised learning algorithm (semantic segmentation) capable of segmenting overlapping chromosomes. The overlapping chromosomes generated, imply only two chromosomes (this is a start). They are obtained by varying the relative positions and orientations of the two chromosomes.

### Project stages
The first stage would to submit one dataset to a semantic segmentation algorithm such segnet. Different  implementations of **Segnet** are available in the current deep-learning frameworks:
     
* [theano+keras+python](https://github.com/pradyu1993/segnet/blob/master/segnet.py)
* [caffe + python](https://github.com/alexgkendall/caffe-segnet)
* [torch](https://github.com/yandex/segnet-torch)

The latter deep-learning framework is supposed to be more efficient than the segnet implementation:

* [Enet + torch + lua](https://github.com/e-lab/ENet-training).

### Dataset
   * **DAPI.tif and Cy3.tif** : 12 bits images of metaphasic chromosomes. The telomeres marking the end of the chromosomes are visible in the Cy3.tif image.The metaphase doesn't contain overlapping chromosomes.
   * **lowres_82146_overlapping_pairs_grey_DAPI-GroundTruth.h5** : 82146 pairs of low resolution (decreased by 4: the overlapping were generated from a DAPI image 16 times smaller than the original image).

   * **LowRes_13434_overlapping_pairs.h5** : 13434 pairs of overlapping chromosomes generated from the two previous images. This dataset is intended to train a supervised learning algorithm to resolve overlapping chromosomes. The dataset is stored as a numpy array and saved in a hdf5 file. Compared to the DAPI and Cy3 images,the resolution was decreased by two.
   * **overlapping_chromosomes_examples.h5**: smaller dataset (~2000 images). The resolution of the images is the same than the DAPI/Cy3 images.
   * **UltraSmall-COCO-Dataset_125_images-json.zip** a very small dataset with 125 images of overlapping chromosomes and their annotation file in COCO format generated online with makesense.ai

## Cytogenetic image multi-channel viewer

The repository now ships a full PyQt5 application dedicated to cytogenetic imaging workflows. Slides are represented by folders that contain numbered metaphases, themselves populated with per-channel sub-directories (e.g. `dapi/`, `cy3/`, `cy5/`, `fitc/`). The viewer arranges the content as a digital *table lumineuse* and exposes dedicated tooling for preprocessing, alignment, segmentation, configuration, and batch execution.

### Launching the application

```bash
python -m viewer /chemin/vers/votre/dataset
```

If no argument is supplied the viewer scans a list of common dataset roots (`jpp21_downloaded_data`, `Raw images/jpp21`, `dataset`, the current folder) and, failing that, generates a small synthetic dataset so the interface remains functional out-of-the-box.

> ℹ️ **Wayland users :** sous GNOME Wayland, forcez `QT_QPA_PLATFORM=wayland` (ou `=xcb`) avant le lancement pour supprimer l'avertissement Qt concernant `XDG_SESSION_TYPE`.

### Interface overview

The main window is split into an image canvas with navigation controls (slide/metaphase selectors and per-channel toggles) and a side panel containing four tabs:

* **Preprocessing** – per-channel white top-hat filtering, background subtraction (median or mode), percentile-based histogram stretching, display mode selection (grayscale, inverted, pseudo-colour) and HSV colour pickers for the latter. Results can be saved on disk, and channel-specific parameters are persisted in `config.json`.
* **Alignment** – manual registration of every spectral component relative to DAPI using arrow buttons or numeric spin boxes. Offsets are stored in `alignment.json` and can be propagated to all metaphases on the slide.
* **Segmentation** – adaptive thresholding (`cv2.adaptiveThreshold`) with adjustable block size and offset, live previews, overlay/contour/filled display modes, undo/redo enabled brush and eraser tools, object size filtering and optional removal of border-touching regions. Masks can be saved as TIFF/PNG files.
* **Batch** – select a dataset root, tick preprocessing/alignment/segmentation steps, and run the pipeline for every slide and metaphase while tracking progress and textual logs.

All parameters edited through the interface are written back to the slide-level `config.json`/`alignment.json` files when the application closes, making it straightforward to resume a session. The same configuration is reused by the batch executor so manual fine-tuning directly feeds the automated pipeline.
