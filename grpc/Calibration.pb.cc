// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Calibration.proto

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "Calibration.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace calibration_grpc {
class ImagesDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<Images> {
} _Images_default_instance_;
class CameraMatrixDefaultTypeInternal : public ::google::protobuf::internal::ExplicitlyConstructed<CameraMatrix> {
} _CameraMatrix_default_instance_;

namespace protobuf_Calibration_2eproto {


namespace {

::google::protobuf::Metadata file_level_metadata[2];

}  // namespace

const ::google::protobuf::uint32 TableStruct::offsets[] = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Images, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Images, image_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Images, phonemodel_),
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CameraMatrix, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CameraMatrix, fx_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CameraMatrix, fy_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CameraMatrix, cx_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(CameraMatrix, cy_),
};

static const ::google::protobuf::internal::MigrationSchema schemas[] = {
  { 0, -1, sizeof(Images)},
  { 6, -1, sizeof(CameraMatrix)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&_Images_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&_CameraMatrix_default_instance_),
};

namespace {

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "Calibration.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 2);
}

}  // namespace

void TableStruct::Shutdown() {
  _Images_default_instance_.Shutdown();
  delete file_level_metadata[0].reflection;
  _CameraMatrix_default_instance_.Shutdown();
  delete file_level_metadata[1].reflection;
}

void TableStruct::InitDefaultsImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::internal::InitProtobufDefaults();
  _Images_default_instance_.DefaultConstruct();
  _CameraMatrix_default_instance_.DefaultConstruct();
}

void InitDefaults() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &TableStruct::InitDefaultsImpl);
}
void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] = {
      "\n\021Calibration.proto\022\020calibration_grpc\"+\n"
      "\006Images\022\r\n\005image\030\001 \003(\014\022\022\n\nphoneModel\030\002 \001"
      "(\t\">\n\014CameraMatrix\022\n\n\002fx\030\001 \001(\001\022\n\n\002fy\030\002 \001"
      "(\001\022\n\n\002cx\030\003 \001(\001\022\n\n\002cy\030\004 \001(\0012]\n\022Calibratio"
      "nService\022G\n\tcalibrate\022\030.calibration_grpc"
      ".Images\032\036.calibration_grpc.CameraMatrix\""
      "\000B-\n\034edu.berkeley.cs.sdb.cellmateB\rCellm"
      "ateProtob\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 296);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "Calibration.proto", &protobuf_RegisterTypes);
  ::google::protobuf::internal::OnShutdown(&TableStruct::Shutdown);
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;

}  // namespace protobuf_Calibration_2eproto


// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int Images::kImageFieldNumber;
const int Images::kPhoneModelFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

Images::Images()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_Calibration_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:calibration_grpc.Images)
}
Images::Images(const Images& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      image_(from.image_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  phonemodel_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.phonemodel().size() > 0) {
    phonemodel_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.phonemodel_);
  }
  // @@protoc_insertion_point(copy_constructor:calibration_grpc.Images)
}

void Images::SharedCtor() {
  phonemodel_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  _cached_size_ = 0;
}

Images::~Images() {
  // @@protoc_insertion_point(destructor:calibration_grpc.Images)
  SharedDtor();
}

void Images::SharedDtor() {
  phonemodel_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void Images::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Images::descriptor() {
  protobuf_Calibration_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_Calibration_2eproto::file_level_metadata[0].descriptor;
}

const Images& Images::default_instance() {
  protobuf_Calibration_2eproto::InitDefaults();
  return *internal_default_instance();
}

Images* Images::New(::google::protobuf::Arena* arena) const {
  Images* n = new Images;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void Images::Clear() {
// @@protoc_insertion_point(message_clear_start:calibration_grpc.Images)
  image_.Clear();
  phonemodel_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

bool Images::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:calibration_grpc.Images)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated bytes image = 1;
      case 1: {
        if (tag == 10u) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadBytes(
                input, this->add_image()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string phoneModel = 2;
      case 2: {
        if (tag == 18u) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_phonemodel()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->phonemodel().data(), this->phonemodel().length(),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "calibration_grpc.Images.phoneModel"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:calibration_grpc.Images)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:calibration_grpc.Images)
  return false;
#undef DO_
}

void Images::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:calibration_grpc.Images)
  // repeated bytes image = 1;
  for (int i = 0; i < this->image_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteBytes(
      1, this->image(i), output);
  }

  // string phoneModel = 2;
  if (this->phonemodel().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->phonemodel().data(), this->phonemodel().length(),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "calibration_grpc.Images.phoneModel");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->phonemodel(), output);
  }

  // @@protoc_insertion_point(serialize_end:calibration_grpc.Images)
}

::google::protobuf::uint8* Images::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic;  // Unused
  // @@protoc_insertion_point(serialize_to_array_start:calibration_grpc.Images)
  // repeated bytes image = 1;
  for (int i = 0; i < this->image_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteBytesToArray(1, this->image(i), target);
  }

  // string phoneModel = 2;
  if (this->phonemodel().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->phonemodel().data(), this->phonemodel().length(),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "calibration_grpc.Images.phoneModel");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->phonemodel(), target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:calibration_grpc.Images)
  return target;
}

size_t Images::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:calibration_grpc.Images)
  size_t total_size = 0;

  // repeated bytes image = 1;
  total_size += 1 *
      ::google::protobuf::internal::FromIntSize(this->image_size());
  for (int i = 0; i < this->image_size(); i++) {
    total_size += ::google::protobuf::internal::WireFormatLite::BytesSize(
      this->image(i));
  }

  // string phoneModel = 2;
  if (this->phonemodel().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->phonemodel());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Images::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:calibration_grpc.Images)
  GOOGLE_DCHECK_NE(&from, this);
  const Images* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const Images>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:calibration_grpc.Images)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:calibration_grpc.Images)
    MergeFrom(*source);
  }
}

void Images::MergeFrom(const Images& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:calibration_grpc.Images)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  image_.MergeFrom(from.image_);
  if (from.phonemodel().size() > 0) {

    phonemodel_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.phonemodel_);
  }
}

void Images::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:calibration_grpc.Images)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Images::CopyFrom(const Images& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:calibration_grpc.Images)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Images::IsInitialized() const {
  return true;
}

void Images::Swap(Images* other) {
  if (other == this) return;
  InternalSwap(other);
}
void Images::InternalSwap(Images* other) {
  image_.UnsafeArenaSwap(&other->image_);
  phonemodel_.Swap(&other->phonemodel_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata Images::GetMetadata() const {
  protobuf_Calibration_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_Calibration_2eproto::file_level_metadata[0];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// Images

// repeated bytes image = 1;
int Images::image_size() const {
  return image_.size();
}
void Images::clear_image() {
  image_.Clear();
}
const ::std::string& Images::image(int index) const {
  // @@protoc_insertion_point(field_get:calibration_grpc.Images.image)
  return image_.Get(index);
}
::std::string* Images::mutable_image(int index) {
  // @@protoc_insertion_point(field_mutable:calibration_grpc.Images.image)
  return image_.Mutable(index);
}
void Images::set_image(int index, const ::std::string& value) {
  // @@protoc_insertion_point(field_set:calibration_grpc.Images.image)
  image_.Mutable(index)->assign(value);
}
void Images::set_image(int index, const char* value) {
  image_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:calibration_grpc.Images.image)
}
void Images::set_image(int index, const void* value, size_t size) {
  image_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:calibration_grpc.Images.image)
}
::std::string* Images::add_image() {
  // @@protoc_insertion_point(field_add_mutable:calibration_grpc.Images.image)
  return image_.Add();
}
void Images::add_image(const ::std::string& value) {
  image_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:calibration_grpc.Images.image)
}
void Images::add_image(const char* value) {
  image_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:calibration_grpc.Images.image)
}
void Images::add_image(const void* value, size_t size) {
  image_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:calibration_grpc.Images.image)
}
const ::google::protobuf::RepeatedPtrField< ::std::string>&
Images::image() const {
  // @@protoc_insertion_point(field_list:calibration_grpc.Images.image)
  return image_;
}
::google::protobuf::RepeatedPtrField< ::std::string>*
Images::mutable_image() {
  // @@protoc_insertion_point(field_mutable_list:calibration_grpc.Images.image)
  return &image_;
}

// string phoneModel = 2;
void Images::clear_phonemodel() {
  phonemodel_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
const ::std::string& Images::phonemodel() const {
  // @@protoc_insertion_point(field_get:calibration_grpc.Images.phoneModel)
  return phonemodel_.GetNoArena();
}
void Images::set_phonemodel(const ::std::string& value) {
  
  phonemodel_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:calibration_grpc.Images.phoneModel)
}
#if LANG_CXX11
void Images::set_phonemodel(::std::string&& value) {
  
  phonemodel_.SetNoArena(
    &::google::protobuf::internal::GetEmptyStringAlreadyInited(), std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:calibration_grpc.Images.phoneModel)
}
#endif
void Images::set_phonemodel(const char* value) {
  
  phonemodel_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:calibration_grpc.Images.phoneModel)
}
void Images::set_phonemodel(const char* value, size_t size) {
  
  phonemodel_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:calibration_grpc.Images.phoneModel)
}
::std::string* Images::mutable_phonemodel() {
  
  // @@protoc_insertion_point(field_mutable:calibration_grpc.Images.phoneModel)
  return phonemodel_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
::std::string* Images::release_phonemodel() {
  // @@protoc_insertion_point(field_release:calibration_grpc.Images.phoneModel)
  
  return phonemodel_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
void Images::set_allocated_phonemodel(::std::string* phonemodel) {
  if (phonemodel != NULL) {
    
  } else {
    
  }
  phonemodel_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), phonemodel);
  // @@protoc_insertion_point(field_set_allocated:calibration_grpc.Images.phoneModel)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// ===================================================================

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int CameraMatrix::kFxFieldNumber;
const int CameraMatrix::kFyFieldNumber;
const int CameraMatrix::kCxFieldNumber;
const int CameraMatrix::kCyFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

CameraMatrix::CameraMatrix()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    protobuf_Calibration_2eproto::InitDefaults();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:calibration_grpc.CameraMatrix)
}
CameraMatrix::CameraMatrix(const CameraMatrix& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&fx_, &from.fx_,
    reinterpret_cast<char*>(&cy_) -
    reinterpret_cast<char*>(&fx_) + sizeof(cy_));
  // @@protoc_insertion_point(copy_constructor:calibration_grpc.CameraMatrix)
}

void CameraMatrix::SharedCtor() {
  ::memset(&fx_, 0, reinterpret_cast<char*>(&cy_) -
    reinterpret_cast<char*>(&fx_) + sizeof(cy_));
  _cached_size_ = 0;
}

CameraMatrix::~CameraMatrix() {
  // @@protoc_insertion_point(destructor:calibration_grpc.CameraMatrix)
  SharedDtor();
}

void CameraMatrix::SharedDtor() {
}

void CameraMatrix::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* CameraMatrix::descriptor() {
  protobuf_Calibration_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_Calibration_2eproto::file_level_metadata[1].descriptor;
}

const CameraMatrix& CameraMatrix::default_instance() {
  protobuf_Calibration_2eproto::InitDefaults();
  return *internal_default_instance();
}

CameraMatrix* CameraMatrix::New(::google::protobuf::Arena* arena) const {
  CameraMatrix* n = new CameraMatrix;
  if (arena != NULL) {
    arena->Own(n);
  }
  return n;
}

void CameraMatrix::Clear() {
// @@protoc_insertion_point(message_clear_start:calibration_grpc.CameraMatrix)
  ::memset(&fx_, 0, reinterpret_cast<char*>(&cy_) -
    reinterpret_cast<char*>(&fx_) + sizeof(cy_));
}

bool CameraMatrix::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:calibration_grpc.CameraMatrix)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // double fx = 1;
      case 1: {
        if (tag == 9u) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &fx_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // double fy = 2;
      case 2: {
        if (tag == 17u) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &fy_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // double cx = 3;
      case 3: {
        if (tag == 25u) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &cx_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // double cy = 4;
      case 4: {
        if (tag == 33u) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &cy_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0 ||
            ::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormatLite::SkipField(input, tag));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:calibration_grpc.CameraMatrix)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:calibration_grpc.CameraMatrix)
  return false;
#undef DO_
}

void CameraMatrix::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:calibration_grpc.CameraMatrix)
  // double fx = 1;
  if (this->fx() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(1, this->fx(), output);
  }

  // double fy = 2;
  if (this->fy() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(2, this->fy(), output);
  }

  // double cx = 3;
  if (this->cx() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(3, this->cx(), output);
  }

  // double cy = 4;
  if (this->cy() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(4, this->cy(), output);
  }

  // @@protoc_insertion_point(serialize_end:calibration_grpc.CameraMatrix)
}

::google::protobuf::uint8* CameraMatrix::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic;  // Unused
  // @@protoc_insertion_point(serialize_to_array_start:calibration_grpc.CameraMatrix)
  // double fx = 1;
  if (this->fx() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(1, this->fx(), target);
  }

  // double fy = 2;
  if (this->fy() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(2, this->fy(), target);
  }

  // double cx = 3;
  if (this->cx() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(3, this->cx(), target);
  }

  // double cy = 4;
  if (this->cy() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(4, this->cy(), target);
  }

  // @@protoc_insertion_point(serialize_to_array_end:calibration_grpc.CameraMatrix)
  return target;
}

size_t CameraMatrix::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:calibration_grpc.CameraMatrix)
  size_t total_size = 0;

  // double fx = 1;
  if (this->fx() != 0) {
    total_size += 1 + 8;
  }

  // double fy = 2;
  if (this->fy() != 0) {
    total_size += 1 + 8;
  }

  // double cx = 3;
  if (this->cx() != 0) {
    total_size += 1 + 8;
  }

  // double cy = 4;
  if (this->cy() != 0) {
    total_size += 1 + 8;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void CameraMatrix::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:calibration_grpc.CameraMatrix)
  GOOGLE_DCHECK_NE(&from, this);
  const CameraMatrix* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const CameraMatrix>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:calibration_grpc.CameraMatrix)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:calibration_grpc.CameraMatrix)
    MergeFrom(*source);
  }
}

void CameraMatrix::MergeFrom(const CameraMatrix& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:calibration_grpc.CameraMatrix)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.fx() != 0) {
    set_fx(from.fx());
  }
  if (from.fy() != 0) {
    set_fy(from.fy());
  }
  if (from.cx() != 0) {
    set_cx(from.cx());
  }
  if (from.cy() != 0) {
    set_cy(from.cy());
  }
}

void CameraMatrix::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:calibration_grpc.CameraMatrix)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void CameraMatrix::CopyFrom(const CameraMatrix& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:calibration_grpc.CameraMatrix)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool CameraMatrix::IsInitialized() const {
  return true;
}

void CameraMatrix::Swap(CameraMatrix* other) {
  if (other == this) return;
  InternalSwap(other);
}
void CameraMatrix::InternalSwap(CameraMatrix* other) {
  std::swap(fx_, other->fx_);
  std::swap(fy_, other->fy_);
  std::swap(cx_, other->cx_);
  std::swap(cy_, other->cy_);
  std::swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata CameraMatrix::GetMetadata() const {
  protobuf_Calibration_2eproto::protobuf_AssignDescriptorsOnce();
  return protobuf_Calibration_2eproto::file_level_metadata[1];
}

#if PROTOBUF_INLINE_NOT_IN_HEADERS
// CameraMatrix

// double fx = 1;
void CameraMatrix::clear_fx() {
  fx_ = 0;
}
double CameraMatrix::fx() const {
  // @@protoc_insertion_point(field_get:calibration_grpc.CameraMatrix.fx)
  return fx_;
}
void CameraMatrix::set_fx(double value) {
  
  fx_ = value;
  // @@protoc_insertion_point(field_set:calibration_grpc.CameraMatrix.fx)
}

// double fy = 2;
void CameraMatrix::clear_fy() {
  fy_ = 0;
}
double CameraMatrix::fy() const {
  // @@protoc_insertion_point(field_get:calibration_grpc.CameraMatrix.fy)
  return fy_;
}
void CameraMatrix::set_fy(double value) {
  
  fy_ = value;
  // @@protoc_insertion_point(field_set:calibration_grpc.CameraMatrix.fy)
}

// double cx = 3;
void CameraMatrix::clear_cx() {
  cx_ = 0;
}
double CameraMatrix::cx() const {
  // @@protoc_insertion_point(field_get:calibration_grpc.CameraMatrix.cx)
  return cx_;
}
void CameraMatrix::set_cx(double value) {
  
  cx_ = value;
  // @@protoc_insertion_point(field_set:calibration_grpc.CameraMatrix.cx)
}

// double cy = 4;
void CameraMatrix::clear_cy() {
  cy_ = 0;
}
double CameraMatrix::cy() const {
  // @@protoc_insertion_point(field_get:calibration_grpc.CameraMatrix.cy)
  return cy_;
}
void CameraMatrix::set_cy(double value) {
  
  cy_ = value;
  // @@protoc_insertion_point(field_set:calibration_grpc.CameraMatrix.cy)
}

#endif  // PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace calibration_grpc

// @@protoc_insertion_point(global_scope)
