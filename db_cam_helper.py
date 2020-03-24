# pip install sqlalchemy psycopg2-binary
# sudo apt-get install build-dep python-psycopg2
# pip install --upgrade onvif_zeep sqlalchemy psycopg2-binary

import os
import sys
import sqlalchemy as db
from datetime import datetime
from urllib.parse import quote
from threading import Thread
from onvif import *

def get_camera_stream_uri(ip, usr="admin", psk="password", wsdl_location="/etc/onvif/wsdl/"):
	# https://www.onvif.org/onvif/ver10/media/wsdl/media.wsdl#op.GetStreamUri
	cam_control = ONVIFCamera(ip, 80, usr, psk, wsdl_location)
	media_control = cam_control.create_media_service()
	cam_config = media_control.GetProfiles()[0].token
	stream_uri_obj = media_control.create_type("GetStreamUri")
	stream_uri_obj.ProfileToken = cam_config
	stream_uri_obj.StreamSetup = {"Stream" : "RTP-Unicast", "Transport" : {"Protocol" : "RTSP"}}
	connection_data = media_control.ws_client.GetStreamUri(stream_uri_obj.StreamSetup, stream_uri_obj.ProfileToken)
	return connection_data


def event_log_dtl_writer(engine, data_dict):
	table_name = '"CV_Analytics".event_log_dtl'
	if engine == None or data_dict == {}:
		return None
	with engine.connect() as link_to_db:
		seq = None
		q = link_to_db.execute(f"SELECT max(seq) from {table_name}")
		if q.rowcount > 0 :
			seq = q.fetchall()[0].values()[0] + 1
		else:
			seq = 1
		data_dict["seq"] = seq
		data_dict["id"] = seq
		insert_query = f"INSERT INTO {table_name} ({','.join(data_dict.keys())}) VALUES {tuple(data_dict.values())}"
		link_to_db.execute(insert_query)

def label_loader(engine, seq_dict, label_template):
	# label_loader(engine, label_dict, label_template)
	table_name = '"CV_Analytics".label'
	if engine == None or seq_dict == {}:
		return None
	with engine.connect() as link_to_db:
		for key in seq_dict:
			seq = None
			new  = True
			q = link_to_db.execute(f"SELECT max(seq) from {table_name}")
			if q.rowcount > 0 :
				q2 = link_to_db.execute(f"SELECT * from {table_name} WHERE created_by={label_template['created_by']}")
				rows = q2.fetchall()
				for row in rows:
					if key in row.values():
						new = False
						break
				if new == False:
					continue
				seq = q.fetchall()[0].values()[0] + 1
			else :
				seq =  list(seq_dict[key].keys())[0]
			label_template["seq"] = seq
			label_template["id"] = seq
			label_template["label_code"] = key
			label_template["label_name"] = seq_dict[key][list(seq_dict[key].keys())[0]]
			label_template["created_date"] = str(datetime.now())
			insert_query = f"INSERT INTO {table_name} ({','.join(label_template.keys())}) VALUES {tuple(label_template.values())}"
			link_to_db.execute(insert_query)

def inference_engine_loader(engine, inference_engine_dict, status):
	table_name = '"CV_Analytics".inference_engine'
	if engine == None or inference_engine_dict == {}:
		return None
	with engine.connect() as link_to_db:
		seq = None
		q = link_to_db.execute(f"SELECT seq from {table_name} WHERE created_by={inference_engine_dict['created_by']}")
		if q.rowcount > 0 :
			query = f"UPDATE {table_name} SET current_flag={status}, updated_date='{datetime.now().replace(tzinfo=None)}', updated_by={inference_engine_dict['created_by']} WHERE created_by={inference_engine_dict['created_by']}"
		else :
			q = link_to_db.execute(f"SELECT max(seq) from {table_name}")
			if q.rowcount > 0 :
			# entry not found, so insert
				seq = q.fetchall()[0].values()[0] + 1
			else:
				seq = 1
			inference_engine_dict["seq"] = seq
			inference_engine_dict["id"] = seq
			inference_engine_dict["created_date"] = str(datetime.now())
			query = f"INSERT INTO {table_name} ({','.join(inference_engine_dict.keys())}) VALUES {tuple(inference_engine_dict.values())}"
		link_to_db.execute(query)


def generate_db_engine(creds):
	if creds == {}:
		return None
	engine = db.create_engine(f'postgresql://{creds["usr"]}:{quote(creds["psk"])}@{creds["ipp"]}/{creds["dbn"]}')
	return engine

model_id, current_flag, active_flag, delete_flag  = 11, 1, 1, 0
model_config_name = "config.json"

label_template = {
	"seq" : None, 
	"id" : None, 
	"label_code" : None, 
	"label_name" : None, 
	"created_by" : model_id, 
	"current_flag" : current_flag, 
	"active_flag" : active_flag, 
	"delete_flag" : delete_flag	
}

label_dict = {
	"NVL" : {5 : "No Violation"},
	"VLO" : {6 : "Violation"}, 
}


inference_engine_dict = {
	"model_id" : model_id,
	"model_vrsn_number" : 1,
	"model_name" : "PPE Detect",
	"model_path" : os.path.abspath(sys.argv[0]),
	"backbone_name" : "yolo v3",
	"model_weight_format" : ".pb, .h5",
	"model_config_name" : model_config_name,
	"model_config_path" : os.path.abspath(model_config_name),
	"model_preprocess_input_shape" : "416, 416",
	"model_framework" : "tf",
	"created_by" : 11,
	"current_flag" : current_flag, 
	"active_flag" : active_flag,
	"delete_flag" : delete_flag
}

creds = {
	"usr" : os.environ["db_usr"], 
	"psk" : os.environ["db_psk"], 
	"dbn" : os.environ["db_name"], 
	"ipp" : os.environ["db_ipp"] # to be replaced with env variables
}

