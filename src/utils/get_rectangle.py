import numpy
def get_rectangle_multiple_positions(points,width,length,yaw):
    points = numpy.transpose(points,(1,0))

    rotMat = numpy.array([[numpy.cos(yaw),-numpy.sin(yaw)],[numpy.sin(yaw),numpy.cos(yaw)]])
    rotMat = numpy.transpose(rotMat,(2,0,1))
    temp_1 = numpy.array([ length,-width])
    temp_2 = numpy.array([ length, width])
    temp_3 = -temp_1
    temp_4 = -temp_2
    points_out_1 = numpy.matmul(rotMat,temp_1) + points
    points_out_2 = numpy.matmul(rotMat,temp_2) + points
    points_out_3 = numpy.matmul(rotMat,temp_3) + points
    points_out_4 = numpy.matmul(rotMat,temp_4) + points

    points_out = numpy.stack((points_out_1,points_out_2,points_out_3,points_out_4),axis=1)
    return points_out