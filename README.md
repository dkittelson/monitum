# Monitum: Wildlife Conservation

Monitum is a machine learning project that identifies at-risk species and forecasts extinction threats over the next decade. By analyzing spatial data on species distributions, climate change, human pressure, and conservation protection, the system calculates a "Resilience Gap" to find "forgotten species"—those facing high vulnerability with inadequate protection. The ML models predict future IUCN Red List categories and identify conservation priority gaps.

## Datasets

- **IUCN Red List Spatial Data** - Global species range maps for mammals, amphibians, and reptiles with current threat status  
  [https://www.iucnredlist.org/resources/spatial-data-download](https://www.iucnredlist.org/resources/spatial-data-download)

- **WorldClim v2.1** - High-resolution climate data for baseline (1970-2000) and future projections (2041-2060)  
  [https://www.worldclim.org/data/worldclim21.html](https://www.worldclim.org/data/worldclim21.html)

- **Human Footprint Index** - Global maps of human pressure intensity (0-50 scale) for 2010 and 2020 from NASA SEDAC  
  [https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint)

- **WDPA (Protected Areas)** - World Database on Protected Areas with polygons for all terrestrial and marine protected zones  
  [https://www.protectedplanet.net/en/thematic-areas/wdpa](https://www.protectedplanet.net/en/thematic-areas/wdpa)

- **Hansen Forest Loss** - Annual forest cover loss data (2000-2023) accessed via Google Earth Engine API  
  [https://earthenginepartners.appspot.com/science-2013-global-forest](https://earthenginepartners.appspot.com/science-2013-global-forest)



## Pipeline

- **Extract Forest Loss:** Firstly, we must process our Hansen Forest Loss data. To do this, we iterate through each species and convert their species boundaries from IUCN geometry to Earth Engine (Where Hansen's data lives). Then, we classify forest as >= 10% canopy and forest loss by Hansen's "loss" column (1 for loss, 0 for none) from 2001-2023. For both forest and loss, we multiply their binary masks by pixel area (900m^2). Consequently, we shrink the region of interest to the species boundaries, discarding all other pixels outside. We then sum up the amount of forest and loss, and convert to km^2. We then calculate the percentage of lost forest and the annual loss rate. 