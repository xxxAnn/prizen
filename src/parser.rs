use std::{fs::{File, self}, path::Path, io::Read, io::Write};

const NEW_LAYER: u8 = 0b00000001;
const NEW_NODE:  u8 = 0b00000010;
const META:      u8 = 0b00000011;

const LINEAR_F:  u8 = 0b00000001;
const AFFINE_F:  u8 = 0b00000010;


// Byte Structure
// (The lines are for clarity)
// number_of_layers
// (NEW_LAYER number_of_nodes (NEW_NODE node_type number_of_weights bias (weight)*number_of_weights)*number_of_nodes)*number_of_layers
// META number_of_input_nodes cst_size cst alpha
// EXP number_of_points ((value)*number_of_input_nodes)*number_of_points (value)*number_of_points)
// all vars are float: [u8; 4]
// (x)*b means (x, x, x, ...) b times.

// Structure
// layer
// node_type_name bias [... weights]
// meta
// input number_of_input_nodes
// alpha learning_rate
// cst cost_function_type
// exp {[... (... values)], [... values]}

fn create_byte_vec<T: AsRef<Path>>(p: T) -> Vec<u8> {
    let f = fs::read_to_string(p).expect("Couldn't read file");
    let lines = f.lines().into_iter().rev().collect::<Vec<&str>>();
    let mut fnal: Vec<u8> = Vec::new();
    let mut meta = Vec::new();
    let mut layers = Vec::new();
    let mut current_value = Vec::new();
    let mut k = true;
    for l in lines {   
        if l.to_lowercase().starts_with("meta") {
            k = false;
        } else {
            if k {
                meta.push(l);
            } else {
                if l.to_lowercase().starts_with("layer") {
                    current_value.reverse();
                    layers.push(current_value);
                    current_value = Vec::new();
                } else {
                    current_value.push(l);
                }
                
            }
        }
    }
    meta.reverse();
    layers.reverse();
    println!("{:?} \n {:?}", meta, layers);
    // Parsing layers
    fnal.append(&mut (layers.len() as f32).to_be_bytes().to_vec()); // adds number of layers bytes
    for layer in layers.iter() {
        fnal.push(NEW_LAYER);
        fnal.append(&mut (layers.len() as f32).to_be_bytes().to_vec());
        for node in layer {
            let first  = node.split_whitespace().next().unwrap();
            fnal.push(match first.to_lowercase().as_str() {
                "linearnode" => LINEAR_F,
                "affinenode" => AFFINE_F,
                _ => panic!("Invalid node name {}", layer[0])
            });
        }
        
    }
    //println!("{:?}", layers);
    vec![]
}

#[test]
fn test_byte_vec() {
    create_byte_vec("model.prcs");
}