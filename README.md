ArchiGAN
===

Following the work of
[ArchiGAN](https://developer.nvidia.com/blog/archigan-generative-stack-apartment-building-design/),
this repository provides scripts for creating datasets for training image translation models via 
[CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


Data Sources
===

[CVC-FP](http://dag.cvc.uab.es/resources/floorplans/)

[CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k)

GIS data from Boston buildings

	1. [Building footprints](https://www.arcgis.com/home/item.html?id=c423eda7a64b49c98a9ebdf5a6b7e135)
	2. [Parcel footprints](https://data.boston.gov/dataset/parcels-2016-data-full/resource/d53d8e93-034d-4dd0-b59f-8634f4df3a71)


Prepared Sources
===

[Repartition model (stage II of original pipeline) pix2pix dataset made from CVC-FP](https://github.com/ctogle/archigan/tree/master/prepared/gt_pix2pix_ABs)
