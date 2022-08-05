import numpy

def crop_numpy_meter(image, size_meter_in, size_meter_out, center_location_out):
    resolution = size_meter_in[::-1]/image.shape[0:]
    size_pixel_out = size_meter_out[::-1]/resolution
    pixel_min = numpy.array([0,0])
    if center_location_out is None:
        pixel_min[0] = ((image.shape[0]/2.0)-( size_pixel_out[0]/2.0)).astype(numpy.int)
        pixel_min[1] = ((image.shape[1]/2.0)-(-size_pixel_out[1]/2.0)).astype(numpy.int)
    else:
        pixel_min[0] = ((image.shape[0]/2.0)-((size_meter_out[1]-center_location_out[1])/resolution[0])).astype(numpy.int)
        pixel_min[1] = ((image.shape[1]/2.0)-(                   center_location_out[0] /resolution[1])).astype(numpy.int)

    input_size = image.shape[0:]
    pixel_max = (pixel_min+size_pixel_out).astype(numpy.int)
    if numpy.any(pixel_min<0) or numpy.any(pixel_max>input_size):
        cropped_image = numpy.array([])
        print('Not yet implemented')
    else:
        cropped_image = image[pixel_min[0]:pixel_max[0],pixel_min[1]:pixel_max[1]]
    return cropped_image