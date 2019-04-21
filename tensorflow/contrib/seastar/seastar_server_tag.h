#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_TAG_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_TAG_H_

#include "core/channel.hh"
#include "core/packet_queue.hh"
#include "core/temporary_buffer.hh"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_service_method.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"

#include <functional>

namespace tensorflow {
typedef std::function<void(const Status&)> StatusCallback;
class SeastarWorkerService;
class SeastarTensorResponse;
class SeastarServerTag;

void InitSeastarServerTag(protobuf::Message* request,
			  protobuf::Message* response,
			  SeastarServerTag* tag);

void InitSeastarServerTag(protobuf::Message* request,
			  SeastarTensorResponse* response,
			  SeastarServerTag* tag,
                          StatusCallback clear);

void InitSeastarServerTag(protobuf::Message* request,
			  SeastarFuseTensorResponse* response,
			  SeastarServerTag* tag,
                          StatusCallback clear);

class SeastarServerTag {
 public:
  // Server Header struct 32B:
  // |ID:8B|tag_id:8B|method:4B|status:2B|err_msg_len:2B|body_len:8B|err_msg...|
  static const uint64_t HEADER_SIZE = 32;
  using HandleRequestFunction = void (SeastarWorkerService::*)(SeastarServerTag*);
  SeastarServerTag(seastar::channel* seastar_channel,
                   SeastarWorkerService* seastar_worker_service);

  virtual ~SeastarServerTag();

  // Called by seastar engine, call the handler.
  void RecvReqDone(Status s);

  // Called by seastar engine.
  void SendRespDone();

  void ProcessDone(Status s);

  uint64_t GetRequestBodySize();

  char* GetRequestBodyBuffer();

  void StartResp();
  void StartRespWithTensors();

  void InitFuse(int32_t fuse_count); 

 private:
  friend class SeastarTagFactory;
  seastar::user_packet* ToUserPacket();
  std::vector<seastar::user_packet*> ToUserPacketWithTensors();

 public:
  SeastarBuf req_body_buf_;
  SeastarBuf resp_header_buf_;
  SeastarBuf resp_body_buf_;
  int32_t fuse_count_;
  std::vector<SeastarBuf> resp_message_bufs_;
  std::vector<SeastarBuf> resp_tensor_bufs_;

  SeastarWorkerServiceMethod method_;

  seastar::channel* seastar_channel_;
  int64_t client_tag_id_;
  
  // Used to serialize and send response data.
  StatusCallback send_resp_;
  StatusCallback clear_;
  int16_t status_;
  SeastarWorkerService* seastar_worker_service_;
};

} // end of namespace tensorflow

#endif // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_TAG_H_
