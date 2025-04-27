# Data Directory

The following data directories belong here:
```
├── berkeley_natural_images
├── brain_tumor
├── brain_ventricles
└── retina
```

As some background info, I inherited the datasets from a graduated member of the lab when I worked on this project. These datasets are already preprocessed by the time I had them. For reproducibility, I have included the `berkeley_natural_images`, `brain_tumor` and `retina` datasets in `zip` format in this directory. The `brain_ventricles` dataset exceeds the GitHub size limits, and can be found on [Google Drive](https://drive.google.com/file/d/1TB5Zu3J4UbEleJUuNf-h1AymOn1jOoQe/view?usp=sharing).

Also, though not used in the paper, I recently found the full (likely unprocessed) NifTI files for the brain tumor dataset too, which I also include in this directory. I believe the brain_tumor dataset used is some of the slices from `brain_tumor_nifti`.

Please be mindful that these datasets are relatively small in sample size. If big sample size is a requirement, you can look into bigger datasets such as the BraTS challenge.
