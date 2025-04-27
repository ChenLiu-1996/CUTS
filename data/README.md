# Data Directory

The following data directories belong here:
```
├── berkeley_natural_images
├── brain_tumor
├── brain_ventricles
└── retina
```

As some background info, I inherited the datasets from a graduated member of the lab when I worked on this project. These datasets are small in sample size N and already preprocessed by the time I had them. For reproducibility, I have included the `berkeley_natural_images`, `brain_tumor` and `retina` datasets in `zip` format in this directory. The `brain_ventricles` dataset exceeds the GitHub size limits, and can be made available upon reasonable request.

Also, though not used in the paper, I recently found the full (likely unprocessed) NifTI files for the brain tumor dataset too, which I also include in this directory. I believe the brain_tumor dataset used is some of the slices from brain_tumor_nifti.

Again, these datasets are relatively small in sample size. You can look into bigger datasets such as the BraTS challenge if big N is a requirement.