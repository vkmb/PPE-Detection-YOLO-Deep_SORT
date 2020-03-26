from db_cam_helper import *

engine = generate_db_engine(creds)

# get confifuration
seq, operating_unit_id, inference_engine_id, label_id = ou_inference_loader(engine)
# load label meta data
label_dict = label_loader(engine, label_id)
# model metadata 
inference_engine_dict = inference_engine_loader(engine, inference_engine_id)
# load operating unit metadata
operating_unit_dict = operating_unit_loader(engine, operating_unit_id)
# run model for the given label , op unit123 

# ['seq', 'id', 'frame_name', 'frame_stored_location', 'frame_size', 'latitude', 'longitude', 'frame_stored_encoding', 'frame_local_epoch', 'frame_local_timestamp', 'frame_local_time_zone', 'created_date', 'created_by', 'active_flag', 'current_flag', 'delete_flag']

# logging
frame_dict = {}
time = datetime.now().strftime("%d/%M/%Y %I:%M:%S %p")
file_name = f'{operating_unit_id} {inference_engine_id} {label_id} {time}'
frame_dict['frame_name'] = file_name
frame_dict['frame_stored_location'] = os.path.abspath(file_name+".jpg")
frame_dict['frame_stored_encoding'] = "JPG"
frame_dict['frame_local_timestamp'] = f"\'{time}\'"
frame_dict['frame_local_time_zone'] = 'IST'
# frame_dict['frame_size'] = image.shape
frame_dict['created_by'] = inference_engine_id
frame_dict['created_date'] =  f"\'{time}\'"
frame_dict['active_flag'] = active_flag
frame_dict['current_flag'] = current_flag
frame_dict['delete_flag'] = delete_flag
frame_id = frame_writer(engine, frame_dict)
object_dtl_dict = {}
object_xmin, object_ymin, object_xmax, object_ymax, label_object_pred_threshold, label_object_pred_confidence = None, None, None, None, None, None
# ['seq', 'id', 'frame_id', 'object_loc_id', 'label_id', 'object_xmin', 'object_ymin', 'object_xmax', 'object_ymax', 'label_object_pred_threshold', 'label_object_pred_confidence', 'created_by', 'created_date', 'updated_by', 'updated_date', 'active_flag', 'current_flag', 'delete_flag']
object_dtl_dict['frame_id'] = frame_id
object_dtl_dict['object_loc_id'] = operating_unit_id
object_dtl_dict['label_id'] = label_id
object_dtl_dict['created_by'] = inference_engine_id
object_dtl_dict['created_date'] =  f"\'{time}\'"
object_dtl_dict['active_flag'] = active_flag
object_dtl_dict['current_flag'] = current_flag
object_dtl_dict['delete_flag'] = delete_flag
object_dtl_dict['object_xmin'] = object_xmin
object_dtl_dict['object_ymin'] = object_ymin
object_dtl_dict['object_xmax'] = object_xmax
object_dtl_dict['object_ymax'] = object_ymax
object_dtl_dict['label_object_pred_threshold'] = label_object_pred_threshold
object_dtl_dict['label_object_pred_confidence'] = label_object_pred_confidence

object_dtl_id = object_dtl_writer(engine, object_dtl_dict)

# ['seq', 'video_id', 'inference_engine_id', 'operating_unit_id', 'label_id', 'frame_id', 'event_processed_local_time', 'event_processed_time_zone', 'event_processed_epoch', 'event_flag', 'created_date', 'created_by', 'update_date', 'updated_by', 'active_flag', 'current_flag', 'delete_flag', 'operating_unit_seq', 'label_seq', 'video_dtl_seq']