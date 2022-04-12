#!/bin/bash
cargo test --features c-headers -- generate_headers
cargo build --release --target aarch64-apple-ios
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-apple-ios
cargo build --release --target aarch64-apple-ios-sim
lipo -create \
  target/x86_64-apple-darwin/release/libhora_c.a \
  target/aarch64-apple-darwin/release/libhora_c.a \
  -output libhora_c_macos.a
lipo -create \
  target/x86_64-apple-ios/release/libhora_c.a \
  target/aarch64-apple-ios-sim/release/libhora_c.a \
  -output libhora_c_iossimulator.a
rm -rf Hora.xcframework
xcodebuild -create-xcframework \
  -library ./libhora_c_macos.a \
  -headers ./include/ \
  -library ./libhora_c_iossimulator.a \
  -headers ./include/ \
  -library ./target/aarch64-apple-ios/release/libhora_c.a \
  -headers ./include/ \
  -output Hora.xcframework
