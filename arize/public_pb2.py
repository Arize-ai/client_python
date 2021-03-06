# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: public.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='public.proto',
  package='public',
  syntax='proto3',
  serialized_options=b'\n\022com.arize.protocolZ9github.com/Arize-ai/arize/go/pkg/receiver/protocol/public',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0cpublic.proto\x12\x06public\x1a\x1fgoogle/protobuf/timestamp.proto\"\x81\x01\n\nBulkRecord\x12\x18\n\x10organization_key\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x15\n\rmodel_version\x18\x03 \x01(\t\x12\x1f\n\x07records\x18\x05 \x03(\x0b\x32\x0e.public.RecordJ\x04\x08\x04\x10\x05R\ttimestamp\"\xa8\x02\n\x06Record\x12\x18\n\x10organization_key\x18\x01 \x01(\t\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x15\n\rprediction_id\x18\x03 \x01(\t\x12(\n\nprediction\x18\x04 \x01(\x0b\x32\x12.public.PredictionH\x00\x12 \n\x06\x61\x63tual\x18\x05 \x01(\x0b\x32\x0e.public.ActualH\x00\x12\x39\n\x13\x66\x65\x61ture_importances\x18\x06 \x01(\x0b\x32\x1a.public.FeatureImportancesH\x00\x12<\n\x15prediction_and_actual\x18\x07 \x01(\x0b\x32\x1b.public.PredictionAndActualH\x00\x42\x16\n\x14prediction_or_actual\"i\n\x0b\x42\x61tchRecord\x12(\n\x0b\x65nvironment\x18\x01 \x01(\x0e\x32\x13.public.Environment\x12\x10\n\x08\x62\x61tch_id\x18\x02 \x01(\t\x12\x1e\n\x06record\x18\x03 \x01(\x0b\x32\x0e.public.Record\"6\n\x10ScoreCategorical\x12\x13\n\x0b\x63\x61tegorical\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x01\"\x82\x01\n\x05Label\x12\x10\n\x06\x62inary\x18\x01 \x01(\x08H\x00\x12\x15\n\x0b\x63\x61tegorical\x18\x02 \x01(\tH\x00\x12\x11\n\x07numeric\x18\x03 \x01(\x01H\x00\x12\x35\n\x11score_categorical\x18\x04 \x01(\x0b\x32\x18.public.ScoreCategoricalH\x00\x42\x06\n\x04\x64\x61ta\"\xe4\x01\n\nPrediction\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x15\n\rmodel_version\x18\x02 \x01(\t\x12\x1c\n\x05label\x18\x03 \x01(\x0b\x32\r.public.Label\x12\x32\n\x08\x66\x65\x61tures\x18\x04 \x03(\x0b\x32 .public.Prediction.FeaturesEntry\x1a>\n\rFeaturesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x1c\n\x05value\x18\x02 \x01(\x0b\x32\r.public.Value:\x02\x38\x01\"m\n\x05Value\x12\x10\n\x06string\x18\x01 \x01(\tH\x00\x12\r\n\x03int\x18\x02 \x01(\x03H\x00\x12\x10\n\x06\x64ouble\x18\x03 \x01(\x01H\x00\x12)\n\x0bmulti_value\x18\x04 \x01(\x0b\x32\x12.public.MultiValueH\x00\x42\x06\n\x04\x64\x61ta\"\x1c\n\nMultiValue\x12\x0e\n\x06values\x18\x01 \x03(\t\"U\n\x06\x41\x63tual\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x1c\n\x05label\x18\x02 \x01(\x0b\x32\r.public.Label\"\xe6\x01\n\x12\x46\x65\x61tureImportances\x12-\n\ttimestamp\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x15\n\rmodel_version\x18\x02 \x01(\t\x12O\n\x13\x66\x65\x61ture_importances\x18\x03 \x03(\x0b\x32\x32.public.FeatureImportances.FeatureImportancesEntry\x1a\x39\n\x17\x46\x65\x61tureImportancesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\"]\n\x13PredictionAndActual\x12&\n\nprediction\x18\x01 \x01(\x0b\x32\x12.public.Prediction\x12\x1e\n\x06\x61\x63tual\x18\x02 \x01(\x0b\x32\x0e.public.Actual*8\n\x0b\x45nvironment\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0c\n\x08TRAINING\x10\x01\x12\x0e\n\nVALIDATION\x10\x02\x42O\n\x12\x63om.arize.protocolZ9github.com/Arize-ai/arize/go/pkg/receiver/protocol/publicb\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,])

_ENVIRONMENT = _descriptor.EnumDescriptor(
  name='Environment',
  full_name='public.Environment',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TRAINING', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VALIDATION', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1571,
  serialized_end=1627,
)
_sym_db.RegisterEnumDescriptor(_ENVIRONMENT)

Environment = enum_type_wrapper.EnumTypeWrapper(_ENVIRONMENT)
UNKNOWN = 0
TRAINING = 1
VALIDATION = 2



_BULKRECORD = _descriptor.Descriptor(
  name='BulkRecord',
  full_name='public.BulkRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='organization_key', full_name='public.BulkRecord.organization_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_id', full_name='public.BulkRecord.model_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_version', full_name='public.BulkRecord.model_version', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='records', full_name='public.BulkRecord.records', index=3,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=58,
  serialized_end=187,
)


_RECORD = _descriptor.Descriptor(
  name='Record',
  full_name='public.Record',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='organization_key', full_name='public.Record.organization_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_id', full_name='public.Record.model_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prediction_id', full_name='public.Record.prediction_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prediction', full_name='public.Record.prediction', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='actual', full_name='public.Record.actual', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='feature_importances', full_name='public.Record.feature_importances', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='prediction_and_actual', full_name='public.Record.prediction_and_actual', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='prediction_or_actual', full_name='public.Record.prediction_or_actual',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=190,
  serialized_end=486,
)


_BATCHRECORD = _descriptor.Descriptor(
  name='BatchRecord',
  full_name='public.BatchRecord',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='environment', full_name='public.BatchRecord.environment', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='batch_id', full_name='public.BatchRecord.batch_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='record', full_name='public.BatchRecord.record', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=488,
  serialized_end=593,
)


_SCORECATEGORICAL = _descriptor.Descriptor(
  name='ScoreCategorical',
  full_name='public.ScoreCategorical',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='categorical', full_name='public.ScoreCategorical.categorical', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score', full_name='public.ScoreCategorical.score', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=595,
  serialized_end=649,
)


_LABEL = _descriptor.Descriptor(
  name='Label',
  full_name='public.Label',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='binary', full_name='public.Label.binary', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='categorical', full_name='public.Label.categorical', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='numeric', full_name='public.Label.numeric', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score_categorical', full_name='public.Label.score_categorical', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='data', full_name='public.Label.data',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=652,
  serialized_end=782,
)


_PREDICTION_FEATURESENTRY = _descriptor.Descriptor(
  name='FeaturesEntry',
  full_name='public.Prediction.FeaturesEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='public.Prediction.FeaturesEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='public.Prediction.FeaturesEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=951,
  serialized_end=1013,
)

_PREDICTION = _descriptor.Descriptor(
  name='Prediction',
  full_name='public.Prediction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='public.Prediction.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_version', full_name='public.Prediction.model_version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='label', full_name='public.Prediction.label', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='features', full_name='public.Prediction.features', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_PREDICTION_FEATURESENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=785,
  serialized_end=1013,
)


_VALUE = _descriptor.Descriptor(
  name='Value',
  full_name='public.Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='string', full_name='public.Value.string', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='int', full_name='public.Value.int', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='double', full_name='public.Value.double', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='multi_value', full_name='public.Value.multi_value', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='data', full_name='public.Value.data',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=1015,
  serialized_end=1124,
)


_MULTIVALUE = _descriptor.Descriptor(
  name='MultiValue',
  full_name='public.MultiValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='public.MultiValue.values', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1126,
  serialized_end=1154,
)


_ACTUAL = _descriptor.Descriptor(
  name='Actual',
  full_name='public.Actual',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='public.Actual.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='label', full_name='public.Actual.label', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1156,
  serialized_end=1241,
)


_FEATUREIMPORTANCES_FEATUREIMPORTANCESENTRY = _descriptor.Descriptor(
  name='FeatureImportancesEntry',
  full_name='public.FeatureImportances.FeatureImportancesEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='public.FeatureImportances.FeatureImportancesEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='public.FeatureImportances.FeatureImportancesEntry.value', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1417,
  serialized_end=1474,
)

_FEATUREIMPORTANCES = _descriptor.Descriptor(
  name='FeatureImportances',
  full_name='public.FeatureImportances',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='public.FeatureImportances.timestamp', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_version', full_name='public.FeatureImportances.model_version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='feature_importances', full_name='public.FeatureImportances.feature_importances', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_FEATUREIMPORTANCES_FEATUREIMPORTANCESENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1244,
  serialized_end=1474,
)


_PREDICTIONANDACTUAL = _descriptor.Descriptor(
  name='PredictionAndActual',
  full_name='public.PredictionAndActual',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='prediction', full_name='public.PredictionAndActual.prediction', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='actual', full_name='public.PredictionAndActual.actual', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1476,
  serialized_end=1569,
)

_BULKRECORD.fields_by_name['records'].message_type = _RECORD
_RECORD.fields_by_name['prediction'].message_type = _PREDICTION
_RECORD.fields_by_name['actual'].message_type = _ACTUAL
_RECORD.fields_by_name['feature_importances'].message_type = _FEATUREIMPORTANCES
_RECORD.fields_by_name['prediction_and_actual'].message_type = _PREDICTIONANDACTUAL
_RECORD.oneofs_by_name['prediction_or_actual'].fields.append(
  _RECORD.fields_by_name['prediction'])
_RECORD.fields_by_name['prediction'].containing_oneof = _RECORD.oneofs_by_name['prediction_or_actual']
_RECORD.oneofs_by_name['prediction_or_actual'].fields.append(
  _RECORD.fields_by_name['actual'])
_RECORD.fields_by_name['actual'].containing_oneof = _RECORD.oneofs_by_name['prediction_or_actual']
_RECORD.oneofs_by_name['prediction_or_actual'].fields.append(
  _RECORD.fields_by_name['feature_importances'])
_RECORD.fields_by_name['feature_importances'].containing_oneof = _RECORD.oneofs_by_name['prediction_or_actual']
_RECORD.oneofs_by_name['prediction_or_actual'].fields.append(
  _RECORD.fields_by_name['prediction_and_actual'])
_RECORD.fields_by_name['prediction_and_actual'].containing_oneof = _RECORD.oneofs_by_name['prediction_or_actual']
_BATCHRECORD.fields_by_name['environment'].enum_type = _ENVIRONMENT
_BATCHRECORD.fields_by_name['record'].message_type = _RECORD
_LABEL.fields_by_name['score_categorical'].message_type = _SCORECATEGORICAL
_LABEL.oneofs_by_name['data'].fields.append(
  _LABEL.fields_by_name['binary'])
_LABEL.fields_by_name['binary'].containing_oneof = _LABEL.oneofs_by_name['data']
_LABEL.oneofs_by_name['data'].fields.append(
  _LABEL.fields_by_name['categorical'])
_LABEL.fields_by_name['categorical'].containing_oneof = _LABEL.oneofs_by_name['data']
_LABEL.oneofs_by_name['data'].fields.append(
  _LABEL.fields_by_name['numeric'])
_LABEL.fields_by_name['numeric'].containing_oneof = _LABEL.oneofs_by_name['data']
_LABEL.oneofs_by_name['data'].fields.append(
  _LABEL.fields_by_name['score_categorical'])
_LABEL.fields_by_name['score_categorical'].containing_oneof = _LABEL.oneofs_by_name['data']
_PREDICTION_FEATURESENTRY.fields_by_name['value'].message_type = _VALUE
_PREDICTION_FEATURESENTRY.containing_type = _PREDICTION
_PREDICTION.fields_by_name['timestamp'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_PREDICTION.fields_by_name['label'].message_type = _LABEL
_PREDICTION.fields_by_name['features'].message_type = _PREDICTION_FEATURESENTRY
_VALUE.fields_by_name['multi_value'].message_type = _MULTIVALUE
_VALUE.oneofs_by_name['data'].fields.append(
  _VALUE.fields_by_name['string'])
_VALUE.fields_by_name['string'].containing_oneof = _VALUE.oneofs_by_name['data']
_VALUE.oneofs_by_name['data'].fields.append(
  _VALUE.fields_by_name['int'])
_VALUE.fields_by_name['int'].containing_oneof = _VALUE.oneofs_by_name['data']
_VALUE.oneofs_by_name['data'].fields.append(
  _VALUE.fields_by_name['double'])
_VALUE.fields_by_name['double'].containing_oneof = _VALUE.oneofs_by_name['data']
_VALUE.oneofs_by_name['data'].fields.append(
  _VALUE.fields_by_name['multi_value'])
_VALUE.fields_by_name['multi_value'].containing_oneof = _VALUE.oneofs_by_name['data']
_ACTUAL.fields_by_name['timestamp'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_ACTUAL.fields_by_name['label'].message_type = _LABEL
_FEATUREIMPORTANCES_FEATUREIMPORTANCESENTRY.containing_type = _FEATUREIMPORTANCES
_FEATUREIMPORTANCES.fields_by_name['timestamp'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_FEATUREIMPORTANCES.fields_by_name['feature_importances'].message_type = _FEATUREIMPORTANCES_FEATUREIMPORTANCESENTRY
_PREDICTIONANDACTUAL.fields_by_name['prediction'].message_type = _PREDICTION
_PREDICTIONANDACTUAL.fields_by_name['actual'].message_type = _ACTUAL
DESCRIPTOR.message_types_by_name['BulkRecord'] = _BULKRECORD
DESCRIPTOR.message_types_by_name['Record'] = _RECORD
DESCRIPTOR.message_types_by_name['BatchRecord'] = _BATCHRECORD
DESCRIPTOR.message_types_by_name['ScoreCategorical'] = _SCORECATEGORICAL
DESCRIPTOR.message_types_by_name['Label'] = _LABEL
DESCRIPTOR.message_types_by_name['Prediction'] = _PREDICTION
DESCRIPTOR.message_types_by_name['Value'] = _VALUE
DESCRIPTOR.message_types_by_name['MultiValue'] = _MULTIVALUE
DESCRIPTOR.message_types_by_name['Actual'] = _ACTUAL
DESCRIPTOR.message_types_by_name['FeatureImportances'] = _FEATUREIMPORTANCES
DESCRIPTOR.message_types_by_name['PredictionAndActual'] = _PREDICTIONANDACTUAL
DESCRIPTOR.enum_types_by_name['Environment'] = _ENVIRONMENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BulkRecord = _reflection.GeneratedProtocolMessageType('BulkRecord', (_message.Message,), {
  'DESCRIPTOR' : _BULKRECORD,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.BulkRecord)
  })
_sym_db.RegisterMessage(BulkRecord)

Record = _reflection.GeneratedProtocolMessageType('Record', (_message.Message,), {
  'DESCRIPTOR' : _RECORD,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.Record)
  })
_sym_db.RegisterMessage(Record)

BatchRecord = _reflection.GeneratedProtocolMessageType('BatchRecord', (_message.Message,), {
  'DESCRIPTOR' : _BATCHRECORD,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.BatchRecord)
  })
_sym_db.RegisterMessage(BatchRecord)

ScoreCategorical = _reflection.GeneratedProtocolMessageType('ScoreCategorical', (_message.Message,), {
  'DESCRIPTOR' : _SCORECATEGORICAL,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.ScoreCategorical)
  })
_sym_db.RegisterMessage(ScoreCategorical)

Label = _reflection.GeneratedProtocolMessageType('Label', (_message.Message,), {
  'DESCRIPTOR' : _LABEL,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.Label)
  })
_sym_db.RegisterMessage(Label)

Prediction = _reflection.GeneratedProtocolMessageType('Prediction', (_message.Message,), {

  'FeaturesEntry' : _reflection.GeneratedProtocolMessageType('FeaturesEntry', (_message.Message,), {
    'DESCRIPTOR' : _PREDICTION_FEATURESENTRY,
    '__module__' : 'public_pb2'
    # @@protoc_insertion_point(class_scope:public.Prediction.FeaturesEntry)
    })
  ,
  'DESCRIPTOR' : _PREDICTION,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.Prediction)
  })
_sym_db.RegisterMessage(Prediction)
_sym_db.RegisterMessage(Prediction.FeaturesEntry)

Value = _reflection.GeneratedProtocolMessageType('Value', (_message.Message,), {
  'DESCRIPTOR' : _VALUE,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.Value)
  })
_sym_db.RegisterMessage(Value)

MultiValue = _reflection.GeneratedProtocolMessageType('MultiValue', (_message.Message,), {
  'DESCRIPTOR' : _MULTIVALUE,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.MultiValue)
  })
_sym_db.RegisterMessage(MultiValue)

Actual = _reflection.GeneratedProtocolMessageType('Actual', (_message.Message,), {
  'DESCRIPTOR' : _ACTUAL,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.Actual)
  })
_sym_db.RegisterMessage(Actual)

FeatureImportances = _reflection.GeneratedProtocolMessageType('FeatureImportances', (_message.Message,), {

  'FeatureImportancesEntry' : _reflection.GeneratedProtocolMessageType('FeatureImportancesEntry', (_message.Message,), {
    'DESCRIPTOR' : _FEATUREIMPORTANCES_FEATUREIMPORTANCESENTRY,
    '__module__' : 'public_pb2'
    # @@protoc_insertion_point(class_scope:public.FeatureImportances.FeatureImportancesEntry)
    })
  ,
  'DESCRIPTOR' : _FEATUREIMPORTANCES,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.FeatureImportances)
  })
_sym_db.RegisterMessage(FeatureImportances)
_sym_db.RegisterMessage(FeatureImportances.FeatureImportancesEntry)

PredictionAndActual = _reflection.GeneratedProtocolMessageType('PredictionAndActual', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONANDACTUAL,
  '__module__' : 'public_pb2'
  # @@protoc_insertion_point(class_scope:public.PredictionAndActual)
  })
_sym_db.RegisterMessage(PredictionAndActual)


DESCRIPTOR._options = None
_PREDICTION_FEATURESENTRY._options = None
_FEATUREIMPORTANCES_FEATUREIMPORTANCESENTRY._options = None
# @@protoc_insertion_point(module_scope)
