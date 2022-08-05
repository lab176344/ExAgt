import numpy 
def in_range(obj, ego, range):
    # Align the time
    obj_pos = numpy.zeros((2,obj.shape[1]))
    ego_pos = numpy.zeros((2,obj.shape[1]))
    for idx,time_idx in enumerate(obj[7, :].astype(int)):
        obj_pos[:,idx] = obj[0:2, idx]
        time_match = numpy.where(ego[7,:].astype(int)==time_idx)[0][0]
        ego_pos[:,idx] = ego[0:2,time_match]
    
    # Get distance and angle
    distances = numpy.sqrt(numpy.sum((ego_pos-obj_pos)**2,axis=0))
    indicator = numpy.less_equal(distances,range)
    return indicator
