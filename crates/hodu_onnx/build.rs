fn main() {
    println!("cargo:rerun-if-changed=proto/onnx.proto");

    // Use bundled protoc from protobuf-src
    std::env::set_var("PROTOC", protobuf_src::protoc());

    prost_build::Config::new()
        .compile_protos(&["proto/onnx.proto3"], &["proto/"])
        .expect("Failed to compile ONNX proto");
}
