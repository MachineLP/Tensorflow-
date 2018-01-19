产品环境模型部署，创建简单Web APP，用户上传图像，运行Inception模型，实现图像自动分类。

搭建TensorFlow服务开发环境。安装Docker，https://docs.docker.com/engine/installation/ 。用配置文件在本地创建Docker镜像，docker build --pull -t $USER/tensorflow-serving-devel https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/tools/docker/Dockerfile.devel 。镜像运行容器，docker run -v $HOME:/mnt/home -p 9999:9999 -it $USER/tensorflow-serving-devel ，在home目录加载到容器/mnt/home路径，在终端工作。用IDE或编辑器编辑代码，用容器运行构建工具，主机通过9999端口访问，构建服务器。exit命令退出容器终端，停止运行。

TensorFlow服务程序C++写，使用Google的Bazel构建工具。容器运行Bazel。Bazel代码级管理第三方依赖项。Bazel自动下载构建。项目库根目录定义WORKSPACE文件。TensorFlow模型库包含Inception模型代码。

TensorFlow服务在项目作为Git子模块。mkdir ~/serving_example，cd ~/serving_example，git init，git submodule add https://github.com/tensorflow/serving.git ，tf_serving，git submodule update --init --recursive 。

WORKSPACE文件local_repository规则定义第三方依赖为本地存储文件。项目导入tf_workspace规则初始化TensorFlow依赖项。

     workspace(name = "serving")

     local_repository(
         name = "tf_serving",
         path = __workspace_dir__ + "/tf_serving",
     )

     local_repository(
         name = "org_tensorflow",
         path = __workspace_dir__ + "/tf_serving/tensorflow",
     )

     load('//tf_serving/tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
     tf_workspace("tf_serving/tensorflow/", "@org_tensorflow")

     bind(
         name = "libssl",
         actual = "@boringssl_git//:ssl",
     )

     bind(
         name = "zlib",
         actual = "@zlib_archive//:zlib",
     )

     local_repository(
         name = "inception_model",
         path = __workspace_dir__ + "/tf_serving/tf_models/inception",
     )

导出训练好的模型，导出数据流图及变量，给产品用。模型数据流图，必须从占位符接收输入，单步推断计算输出。Inception模型(或一般图像识别模型)，JPEG编码图像字符串输入，与从TFRecord文件读取输入不同。定义输入占位符，调用函数转换占位符表示外部输入为原始推断模型输入格式，图像字符串转换为各分量位于[0, 1]内像素张量，缩放图像尺寸，符合模型期望宽度高度，像素值变换到模型要求区间[-1, 1]内。调用原始模型推断方法，依据转换输入推断结果。

推断方法各参数赋值。从检查点恢复参数值。周期性保存模型训练检查点文件，文件包含学习参数。最后一次保存训练检查点文件包含最后更新模型参数。下去载预训练检查点文件：http://download.tensorflow.org/models/imagenet/inception-v3-2016-03-01.tar.gz 。在Docker容器中，cd /tmp, curl -0 http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz, tar -xzf inception-v3-2016-03-01.tar.gz 。

tensorflow_serving.session_bundle.exporter.Exporter类导出模型。传入保存器实例创建实例，用exporter.classification_signature创建模型签名。指定input_tensor、输出张量。classes_tensor 包含输出类名称列表、模型分配各类别分值(或概率)socres_tensor。类别数多模型，配置指定仅返田大口tf.nntop_k选择类别，模型分配分数降序排列前K个类别。调用exporter.Exporter.init方法签名，export方法导出模型，接收输出路径、模型版本号、会话对象。Exporter类自动生成代码存在依赖，Doker容器内部使用中bazel运行导出器。代码保存到bazel工作区exporter.py。

     import time
     import sys

     import tensorflow as tf
     from tensorflow_serving.session_bundle import exporter
     from inception import inception_model

     NUM_CLASSES_TO_RETURN = 10

     def convert_external_inputs(external_x):
         image = tf.image.convert_image_dtype(tf.image.decode_jpeg(external_x, channels=3), tf.float32)
         images = tf.image.resize_bilinear(tf.expand_dims(image, 0), [299, 299])
         images = tf.mul(tf.sub(images, 0.5), 2)
         return images

     def inference(images):
         logits, _ = inception_model.inference(images, 1001)
         return logits

     external_x = tf.placeholder(tf.string)
     x = convert_external_inputs(external_x)
     y = inference(x)

     saver = tf.train.Saver()

     with tf.Session() as sess:
         ckpt = tf.train.get_checkpoint_state(sys.argv[1])
         if ckpt and ckpt.model_checkpoint_path:
             saver.restore(sess, sys.argv[1] + "/" + ckpt.model_checkpoint_path)
         else:
             print("Checkpoint file not found")
             raise SystemExit

         scores, class_ids = tf.nn.top_k(y, NUM_CLASSES_TO_RETURN)

         classes = tf.contrib.lookup.index_to_string(tf.to_int64(class_ids),
             mapping=tf.constant([str(i) for i in range(1001)]))

         model_exporter = exporter.Exporter(saver)
         signature = exporter.classification_signature(
             input_tensor=external_x, classes_tensor=classes, scores_tensor=scores)
         model_exporter.init(default_graph_signature=signature, init_op=tf.initialize_all_tables())
         model_exporter.export(sys.argv[1] + "/export", tf.constant(time.time()), sess)

一个构建规则BUILD文件。在容器命令运行导出器，cd /mnt/home/serving_example, hazel run:export /tmp/inception-v3 ，依据/tmp/inception-v3提到的检查点文件在/tmp/inception-v3/{currenttimestamp}/创建导出器。首次运行要对TensorFlow编译。load从外部导入protobuf库，导入cc_proto_library规则定义，为proto文件定义构建规则。通过命令bazel run :server 9999 /tmp/inception-v3/export/{timestamp}，容器运行推断服务器。

     py_binary(
         name = "export",
         srcs = [
             "export.py",
         ],
         deps = [
             "@tf_serving//tensorflow_serving/session_bundle:exporter",
             "@org_tensorflow//tensorflow:tensorflow_py",
             "@inception_model//inception",
         ],
     )

     load("@protobuf//:protobuf.bzl", "cc_proto_library")

     cc_proto_library(
         name="classification_service_proto",
         srcs=["classification_service.proto"],
         cc_libs = ["@protobuf//:protobuf"],
         protoc="@protobuf//:protoc",
         default_runtime="@protobuf//:protobuf",
         use_grpc_plugin=1
     )

     cc_binary(
         name = "server",
         srcs = [
             "server.cc",
             ],
         deps = [
             ":classification_service_proto",
             "@tf_serving//tensorflow_serving/servables/tensorflow:session_bundle_factory",
             "@grpc//:grpc++",
             ],
     )

定义服务器接口。TensorFlow服务使用gRPC协议(基于HTTP/2二进制协议)。支持创建服务器和自动生成客户端存根各种语言。在protocol buffer定义服务契约，用于gRPC IDL(接口定义语言)和二进制编码。接收JPEG编码待分类图像字符串输入，返回分数排列推断类别列表。定义在classification_service.proto文件。接收图像、音频片段、文字服务可用可一接口。proto编译器转换proto文件为客户端和服务器类定义。bazel build:classification_service_proto可行构建，通过bazel-genfiles/classification_service.grpc.pb.h检查结果。推断逻辑，ClassificationService::Service接口必须实现。检查bazel-genfiles/classification_service.pb.h查看request、response消息定义。proto定义变成每种类型C++接口。

     syntax = "proto3";

     message ClassificationRequest {
        // bytes input = 1;
        float petalWidth = 1;
        float petalHeight = 2;
        float sepalWidth = 3;
        float sepalHeight = 4;
     };

     message ClassificationResponse {
        repeated ClassificationClass classes = 1;
     };

     message ClassificationClass {
        string name = 1;
        float score = 2;
     }

     service ClassificationService {
        rpc classify(ClassificationRequest) returns (ClassificationResponse);
     }

实现推断服务器。加载导出模型，调用推断方法，实现ClassificationService::Service。导出模型，创建SessionBundle对象，包含完全加载数据流图TF会话对象，定义导出工具分类签名元数据。SessionBundleFactory类创建SessionBundle对象，配置为pathToExportFiles指定路径加载导出模型，返回创建SessionBundle实例unique指针。定义ClassificationServiceImpl，接收SessionBundle实例参数。

加载分类签名，GetClassificationSignature函数加载模型导出元数据ClassificationSignature，签名指定所接收图像真实名称的输入张量逻辑名称，以及数据流图输出张量逻辑名称映射推断结果。将protobuf输入变换为推断输入张量，request参数复制JPEG编码图像字符串到推断张量。运行推断，sessionbundle获得TF会话对象，运行一次，传入输入输出张量推断。推断输出张量变换protobuf输出，输出张量结果复制到ClassificationResponse消息指定形状response输出参数格式化。设置gRPC服务器，SessionBundle对象配置，创建ClassificationServiceImpl实例样板代码。

     #include <iostream>
     #include <memory>
     #include <string>

     #include <grpc++/grpc++.h>

     #include "classification_service.grpc.pb.h"

     #include "tensorflow_serving/servables/tensorflow/session_bundle_factory.h"

     using namespace std;
     using namespace tensorflow::serving;
     using namespace grpc;

     unique_ptr<SessionBundle> createSessionBundle(const string& pathToExportFiles) {
        SessionBundleConfig session_bundle_config = SessionBundleConfig();
        unique_ptr<SessionBundleFactory> bundle_factory;
        SessionBundleFactory::Create(session_bundle_config, &bundle_factory);

        unique_ptr<SessionBundle> sessionBundle;
        bundle_factory->CreateSessionBundle(pathToExportFiles, &sessionBundle);

        return sessionBundle;
     }


     class ClassificationServiceImpl final : public ClassificationService::Service {

       private:
        unique_ptr<SessionBundle> sessionBundle;

       public:
         ClassificationServiceImpl(unique_ptr<SessionBundle> sessionBundle) :
        sessionBundle(move(sessionBundle)) {};

        Status classify(ServerContext* context, const ClassificationRequest* request,
                    ClassificationResponse* response) override {

           ClassificationSignature signature;
           const tensorflow::Status signatureStatus =
             GetClassificationSignature(sessionBundle->meta_graph_def, &signature);

           if (!signatureStatus.ok()) {
              return Status(StatusCode::INTERNAL, signatureStatus.error_message());
           }

           tensorflow::Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
           input.scalar<string>()() = request->input();

           vector<tensorflow::Tensor> outputs;

           const tensorflow::Status inferenceStatus = sessionBundle->session->Run(
              {{signature.input().tensor_name(), input}},
              {signature.classes().tensor_name(), signature.scores().tensor_name()},
              {},
              &outputs);

           if (!inferenceStatus.ok()) {
              return Status(StatusCode::INTERNAL, inferenceStatus.error_message());
           }

           for (int i = 0; i < outputs[0].NumElements(); ++i) {
              ClassificationClass *classificationClass = response->add_classes();
              classificationClass->set_name(outputs[0].flat<string>()(i));
              classificationClass->set_score(outputs[1].flat<float>()(i));
           }

             return Status::OK;

         }
     };


     int main(int argc, char** argv) {

         if (argc < 3) {
            cerr << "Usage: server <port> /path/to/export/files" << endl;
           return 1;
         }

        const string serverAddress(string("0.0.0.0:") + argv[1]);
        const string pathToExportFiles(argv[2]);

        unique_ptr<SessionBundle> sessionBundle = createSessionBundle(pathToExportFiles);

        ClassificationServiceImpl classificationServiceImpl(move(sessionBundle));

         ServerBuilder builder;
         builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
         builder.RegisterService(&classificationServiceImpl);

         unique_ptr<Server> server = builder.BuildAndStart();
         cout << "Server listening on " << serverAddress << endl;

         server->Wait();

         return 0;
     }

通过服务器端组件从webapp访问推断服务。运行Python protocol buffer编译器，生成ClassificationService Python protocol buffer客户端：pip install grpcio cython grpcio-tools, python -m grpc.tools.protoc -I. --python_out=. --grpc_python_out=. classification_service.proto。生成包含调用服务stub classification_service_pb2.py 。服务器接到POST请求，解析发送表单，创建ClassificationRequest对象 。分类服务器设置一个channel，请求提交，分类响应渲染HTML，送回用户。容器外部命令python client.py，运行服务器。浏览器导航http://localhost:8080 访问UI。

     from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

     import cgi
     import classification_service_pb2
     from grpc.beta import implementations

     class ClientApp(BaseHTTPRequestHandler):
         def do_GET(self):
             self.respond_form()

         def respond_form(self, response=""):

             form = """
             <html><body>
             <h1>Image classification service</h1>
             <form enctype="multipart/form-data" method="post">
             <div>Image: <input type="file" name="file" accept="image/jpeg"></div>
             <div><input type="submit" value="Upload"></div>
             </form>
             %s
             </body></html>
             """

             response = form % response

             self.send_response(200)
             self.send_header("Content-type", "text/html")
             self.send_header("Content-length", len(response))
             self.end_headers()
             self.wfile.write(response)

         def do_POST(self):

             form = cgi.FieldStorage(
                 fp=self.rfile,
                 headers=self.headers,
                 environ={
                     'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                 })

             request = classification_service_pb2.ClassificationRequest()
             request.input = form['file'].file.read()

             channel = implementations.insecure_channel("127.0.0.1", 9999)
             stub = classification_service_pb2.beta_create_ClassificationService_stub(channel)
             response = stub.classify(request, 10) # 10 secs timeout

             self.respond_form("<div>Response: %s</div>" % response)


     if __name__ == '__main__':
         host_port = ('0.0.0.0', 8080)
         print "Serving in %s:%s" % host_port
         HTTPServer(host_port, ClientApp).serve_forever()

产品准备，分类服务器应用产品。编译服务器文件复制到容器永久位置，清理所有临时构建文件。容器中，mkdir /opt/classification_server, cd /mnt/home/serving_example, cp -R bazel-bin/. /opt/classification_server, bazel clean 。容器外部，状态提交新Docker镜像，创建记录虚拟文件系统变化快照。容器外，docker ps, dock commit <container id>。图像推送到自己偏好docker服务云，服务。

参考资料：
《面向机器智能的TensorFlow实践》


