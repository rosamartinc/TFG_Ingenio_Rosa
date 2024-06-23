// Primer paso: Crear una nueva Annotation de tipo Rectangle que abarque la zona de interés. 
// A continuación ejecutar el script: 

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData() 

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

if (name.endsWith(".tiff")){ 
    name = name.substring(0, name.length() - 5)
}

// Define output path del proyecto

def pathOutput = buildFilePath("C:/Users/rmart/Desktop/4o/2S/TFG/INGENIO/DATOS", 'parches_proyector', name)
mkdirs(pathOutput)

// Definición de path donde guardar ROI, en caso de ser preciso su extracción completa
/*
def pathOutput_roi = buildFilePath("C:/Users/rmart/Desktop/4o/2S/TFG/INGENIO/DATOS", 'roi_proyectos', name)
mkdirs(pathOutput_roi)
*/

// Define output resolution

double requestedPixelSize = 0.25

// Downsample

double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Creación de ImageServer 

def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) 
    .downsample(downsample) // Elección de server resolution
    .useCellNuclei() // Si se pretendiera extraer máscara de células en vez de núcleos usar:  .useCells() https://www.imagescientist.com/scripting-export-images
    .addLabel('Tumor', 1) 
    .addLabel('Normal', 2)
    .addLabel('Stroma', 3)
    .addUnclassifiedLabel(4)
    .lineThickness(1) 
    .setBoundaryLabel('Anything*', 0) 
    
 def labelServerFalse = labelServer.multichannelOutput(false).build() 

// Exportar región

int i = 0
for (annotation in getAnnotationObjects()) {
   
    i++
    def subcadena = 'Rectangle'
    def cadena = annotation.getROI()
    
    if (cadena.toString().contains(subcadena.toString())) { 
        
        def regionFalse = RegionRequest.createInstance(labelServerFalse.getPath(), downsample, annotation.getROI())
        
        // Extracción de máscara de la ROI, en caso de ser preciso su extracción completa
        /*
        def outputPathF_roi = buildFilePath(pathOutput_roi, name + '-roi.png')
        writeImageRegion(labelServerFalse, regionFalse, outputPathF_roi)
        */
        
        def regionCrop = RegionRequest.createInstance(imageData.getServer().getPath(), downsample, annotation.getROI())

        // Extracción de imagen de la ROI, en caso de ser preciso su extracción completa
        /*
        def img = imageData.getServer().readRegion(regionCrop)
        def outputPathCrop_roi = buildFilePath(pathOutput_roi, name + '-roi.jpg')
        writeImage(img, outputPathCrop_roi)
        */

        // Extracción de parches/tiles de la ROI: imagen y máscara de cada parche

        new TileExporter(imageData)
            .region(regionCrop) // Solo tiles de la parte de la ROI
            .downsample(downsample) // Define export resolution
            .imageExtension('.jpg') // Define file extension 
            .tileSize(540) // Define tamaño parche, en píxels
            .labeledServer(labelServerFalse) // Define the labeled image server to use (i.e. the one we just built)
            .annotatedTilesOnly(true) // If true, solo exporta tiles con anotación presente
            .overlap(164) // Define overlap, en pixel 
            .writeTiles(pathOutput)     
    }
}

print('Done!')


