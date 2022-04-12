use std::collections::HashMap;
use std::sync::Mutex;
use std::os::raw::{c_char};
use std::ffi::{CStr};
use std::slice;
use hora::core::ann_index::SerializableIndex;
use ::safer_ffi::prelude::*;

#[macro_use]
extern crate lazy_static;

trait ANNIndexer:
    hora::core::ann_index::ANNIndex<f64, String>
    + hora::core::ann_index::SerializableIndex<f64, String>
{
}

impl ANNIndexer for hora::index::hnsw_idx::HNSWIndex<f64, String> {}

pub fn metrics_transform(s: &str) -> hora::core::metrics::Metric {
    match s {
        "angular" => hora::core::metrics::Metric::Angular,
        "manhattan" => hora::core::metrics::Metric::Manhattan,
        "dot_product" => hora::core::metrics::Metric::DotProduct,
        "euclidean" => hora::core::metrics::Metric::Euclidean,
        "cosine_similarity" => hora::core::metrics::Metric::CosineSimilarity,
        _ => hora::core::metrics::Metric::Unknown,
    }
}

lazy_static! {
    static ref ANN_INDEX_MANAGER: Mutex<HashMap<String, Box<dyn ANNIndexer>>> =
        Mutex::new(HashMap::new());
}

#[ffi_export]
pub extern fn hora_new_bf_index(
    name: *const c_char,
    dimension: usize,
) {
    let idx_name = cchar_to_string(name);
    let idx_dimension = dimension as usize;

    ANN_INDEX_MANAGER.lock().unwrap().insert(
        idx_name,
        Box::new(hora::index::hnsw_idx::HNSWIndex::<
            f64,
            String,
        >::new(
            idx_dimension,
            &hora::index::hnsw_params::HNSWParams::default(),
        )),
    );
}

fn cchar_to_string(name: *const c_char) -> String {
    let idx_name;
    unsafe {
        idx_name = CStr::from_ptr(name).to_string_lossy().into_owned();
    }
    idx_name
}

#[ffi_export]
pub extern fn hora_add(
    name: *const c_char,
    features: *const f64,
    idx: *const c_char,
    dimension: usize
) {
    let idx_name: String = cchar_to_string(name);
    let idx = cchar_to_string(idx);
    let data_slice = unsafe { slice::from_raw_parts(features as *const f64, dimension) };
    let buf = data_slice.to_vec();

    match &mut ANN_INDEX_MANAGER.lock().unwrap().get_mut(&idx_name) {
        Some(index) => {
            index.add(&buf, idx).unwrap();
        }
        None => {}
    }
}

#[ffi_export]
pub extern fn hora_build(
    name: *const c_char,
    mt: *const c_char,
) -> char_p::Box {
    let idx_name: String = cchar_to_string(name);
    let metric: String = cchar_to_string(mt);

    match &mut ANN_INDEX_MANAGER.lock().unwrap().get_mut(&idx_name) {
        Some(index) => {
            match index.build(metrics_transform(&metric)) {
                Ok(()) => char_p::new("Ok"),
                Err(e) => char_p::new(e),
            }
        }
        None => {
            return char_p::new("No index")
        }
    }
}

#[ffi_export]
pub extern fn hora_search(
    name: *const c_char,
    k: usize,
    features: *const f64,
    dimension: usize
) -> repr_c::Vec<char_p::Box> {
    let idx_name: String = cchar_to_string(name);
    let data_slice = unsafe { slice::from_raw_parts(features, dimension) };
    let buf = data_slice.to_vec();
    let topk = k;

    let mut result: Vec<char_p::Box> = vec![];
    if let Some(index) = ANN_INDEX_MANAGER.lock().unwrap().get(&idx_name) {
        index.search(&buf, topk).iter().for_each( |x| {
            result.push(char_p::new(x.clone() ))
        })

    }
    result.into()
}

#[ffi_export]
pub extern fn hora_load(
    name: *const c_char,
    _file_path: *const c_char,
) {
    let idx_name: String = cchar_to_string(name);
    let file_path: String = cchar_to_string(_file_path);
    ANN_INDEX_MANAGER.lock().unwrap().insert(
        idx_name,
        Box::new(
            hora::index::hnsw_idx::HNSWIndex::<f64, String>::load(&file_path)
            .unwrap(),
        ),
    );
}

#[ffi_export]
pub extern fn hora_dump(
    name: *const c_char,
    _file_path: *const c_char,
) {
    let idx_name: String = cchar_to_string(name);
    let file_path: String = cchar_to_string(_file_path);

    if let Some(index) = ANN_INDEX_MANAGER.lock().unwrap().get_mut(&idx_name) {
        index
            .dump(&file_path)
            .unwrap();
    }
}

/// The following test function is necessary for the header generation.
#[::safer_ffi::cfg_headers]
#[test]
fn generate_headers() -> ::std::io::Result<()> {
    ::safer_ffi::headers::builder()
        .to_file("include/hora.h")?
        .generate()
}

#[test]
fn hora_test() {
    ANN_INDEX_MANAGER.lock().unwrap().insert(
        "test".parse().unwrap(),
        Box::new(hora::index::hnsw_idx::HNSWIndex::<
            f64,
            String,
        >::new(
            8,
            &hora::index::hnsw_params::HNSWParams::<f64>::default(),
        )),
    );
    let buf: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let buf2: Vec<f64> = vec![2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
    let buf3: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let mut result: Vec<String> = vec![];
    match &mut ANN_INDEX_MANAGER.lock().unwrap().get_mut("test") {
        Some(index) => {
            index.add(&buf, "id".to_string()).unwrap();
            index.add(&buf2, "id2".to_string()).unwrap();
            index.add(&buf3, "id3".to_string()).unwrap();
            println!("Added nodes");
            index.build(metrics_transform("euclidean")).unwrap();

            println!("Built\n");
            index.search(&buf, 3).iter().for_each( |x| {
                println!("Found result: {}", x);
                result.push(x.clone())
            })
        }
        None => {}
    }
    for x in result {
        println!("{}", x);
    }
}