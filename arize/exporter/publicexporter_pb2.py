# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: publicexporter.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14publicexporter.proto\x12\x0epublicexporter\x1a\x1fgoogle/protobuf/timestamp.proto\"\x98\x06\n\x15RecordQueryDescriptor\x12\x10\n\x08space_id\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x46\n\x0b\x65nvironment\x18\x03 \x01(\x0e\x32\x31.publicexporter.RecordQueryDescriptor.Environment\x12\x15\n\rmodel_version\x18\x04 \x01(\t\x12\x10\n\x08\x62\x61tch_id\x18\x05 \x01(\t\x12.\n\nstart_time\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12,\n\x08\x65nd_time\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x17\n\x0finclude_actuals\x18\x08 \x01(\x08\x12\x19\n\x11\x66ilter_expression\x18\t \x01(\t\x12^\n\x18similarity_search_params\x18\n \x01(\x0b\x32<.publicexporter.RecordQueryDescriptor.SimilaritySearchParams\x1a\xa0\x02\n\x16SimilaritySearchParams\x12Z\n\nreferences\x18\x01 \x03(\x0b\x32\x46.publicexporter.RecordQueryDescriptor.SimilaritySearchParams.Reference\x12\x1a\n\x12search_column_name\x18\x02 \x01(\t\x12\x11\n\tthreshold\x18\x03 \x01(\x01\x1a{\n\tReference\x12\x15\n\rprediction_id\x18\x01 \x01(\t\x12\x1d\n\x15reference_column_name\x18\x02 \x01(\t\x12\x38\n\x14prediction_timestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"U\n\x0b\x45nvironment\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0c\n\x08TRAINING\x10\x01\x12\x0e\n\nVALIDATION\x10\x02\x12\x0e\n\nPRODUCTION\x10\x03\x12\x0b\n\x07TRACING\x10\x04\x42GZEgithub.com/Arize-ai/arize/go/pkg/flightserver/protocol/publicexporterb\x06proto3')



_RECORDQUERYDESCRIPTOR = DESCRIPTOR.message_types_by_name['RecordQueryDescriptor']
_RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS = _RECORDQUERYDESCRIPTOR.nested_types_by_name['SimilaritySearchParams']
_RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS_REFERENCE = _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS.nested_types_by_name['Reference']
_RECORDQUERYDESCRIPTOR_ENVIRONMENT = _RECORDQUERYDESCRIPTOR.enum_types_by_name['Environment']
RecordQueryDescriptor = _reflection.GeneratedProtocolMessageType('RecordQueryDescriptor', (_message.Message,), {

  'SimilaritySearchParams' : _reflection.GeneratedProtocolMessageType('SimilaritySearchParams', (_message.Message,), {

    'Reference' : _reflection.GeneratedProtocolMessageType('Reference', (_message.Message,), {
      'DESCRIPTOR' : _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS_REFERENCE,
      '__module__' : 'publicexporter_pb2'
      # @@protoc_insertion_point(class_scope:publicexporter.RecordQueryDescriptor.SimilaritySearchParams.Reference)
      })
    ,
    'DESCRIPTOR' : _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS,
    '__module__' : 'publicexporter_pb2'
    # @@protoc_insertion_point(class_scope:publicexporter.RecordQueryDescriptor.SimilaritySearchParams)
    })
  ,
  'DESCRIPTOR' : _RECORDQUERYDESCRIPTOR,
  '__module__' : 'publicexporter_pb2'
  # @@protoc_insertion_point(class_scope:publicexporter.RecordQueryDescriptor)
  })
_sym_db.RegisterMessage(RecordQueryDescriptor)
_sym_db.RegisterMessage(RecordQueryDescriptor.SimilaritySearchParams)
_sym_db.RegisterMessage(RecordQueryDescriptor.SimilaritySearchParams.Reference)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZEgithub.com/Arize-ai/arize/go/pkg/flightserver/protocol/publicexporter'
  _RECORDQUERYDESCRIPTOR._serialized_start=74
  _RECORDQUERYDESCRIPTOR._serialized_end=866
  _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS._serialized_start=491
  _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS._serialized_end=779
  _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS_REFERENCE._serialized_start=656
  _RECORDQUERYDESCRIPTOR_SIMILARITYSEARCHPARAMS_REFERENCE._serialized_end=779
  _RECORDQUERYDESCRIPTOR_ENVIRONMENT._serialized_start=781
  _RECORDQUERYDESCRIPTOR_ENVIRONMENT._serialized_end=866
# @@protoc_insertion_point(module_scope)
