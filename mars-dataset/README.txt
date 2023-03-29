Mars orbital image (HiRISE) labeled data set
--------------------------------------------
Authors: Steven Lu and Kiri L. Wagstaff
Contact: kiri.l.wagstaff@jpl.nasa.gov

This data set contains 3820 landmarks that were extracted from 168
HiRISE images. The landmarks were detected in HiRISE browse
images. For each landmark, we cropped a square bounding box the
included the full extent of the landmark plus a 30-pixel margin to
left, right, top, and bottom. Each cropped image was then resized to
227x227 pixels.

Contents:
- map-proj/: Directory containing individual cropped landmark images
- labels-map-proj.txt: Class labels (ids) for each landmark image
- landmark_mp.py: Python dictionary that maps class ids to semantic
names

Attribution: 
If you use this data set in your own work, please cite this DOI:

10.5281/zenodo.1048301

Please also cite this paper, which provides additional details about
the data set.

Kiri L. Wagstaff, You Lu, Alice Stanboli, Kevin Grimes, Thamme Gowda,
and Jordan Padams. "Deep Mars: CNN Classification of Mars Imagery for
the PDS Imaging Atlas." Proceedings of the Thirtieth Annual Conference
on Innovative Applications of Artificial Intelligence, 2017.

