import numpy
def in_field_of_view(obj, ego, range, angle_range):
    # Align the time
    obj_pos = numpy.zeros((2,obj.shape[1]))
    ego_pos = numpy.zeros((2,obj.shape[1]))
    ego_psi = numpy.zeros((1,obj.shape[1]))
    for idx,time_idx in enumerate(obj[7, :].astype(int)):
        obj_pos[:,idx] = obj[0:2, idx]
        time_match = numpy.where(ego[7,:].astype(int)==time_idx)[0][0]
        ego_pos[:,idx] = ego[0:2,time_match]
        ego_psi[:,idx] = ego[3,time_match]
    
    # Get distance and angle
    distances = numpy.sqrt(numpy.sum((ego_pos-obj_pos)**2,axis=0))
    indicator = numpy.less_equal(distances,range)
    # Check if distance is in range
    # check if angle is in yaw +- angle_range
    angle = numpy.arctan2(obj_pos[1,:]-ego_pos[1,:],obj_pos[0,:]-ego_pos[0,:])
    angle_diff = (ego_psi-angle) % (2*3.14)
    angle_diff[angle_diff>=3.14] -= 2*3.14
    angle_diff = numpy.abs(angle_diff)
    indicator = numpy.bitwise_and(indicator,numpy.less_equal(angle_diff[0],(angle_range/2.0)/180.0*3.14))

    return indicator