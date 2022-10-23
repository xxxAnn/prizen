use std::{fs::{File, self}, path::Path, io::Read, io::Write};

use crate::{LinearNode, AffineNode, Node};

const NEW_LAYER: u8 = 0b00000001;
const NEW_NODE:  u8 = 0b00000010;
const META:      u8 = 0b00000011;

const LINEAR_F:  u8 = 0b00000001;
const AFFINE_F:  u8 = 0b00000010;

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
    let mut r = Vec::new();
    for layer in layers.iter() {
        let mut ls = Vec::new();
        for node in layer {
            let n = node.replace("[", "").replace("]", "");
            let els  = n.split_whitespace().collect::<Vec<&str>>();
            let first = els[0];
            
            let b = els[1].parse::<f64>().unwrap();
            let weights = &els[2..];
            let mut w = Vec::new();
            for x in weights {
                w.push(x.parse::<f64>().unwrap());
            }

            let t: Box<dyn Node> = match first.to_lowercase().as_str() {
                "linearnode" => Box::new(LinearNode { w, b, needed: weights.len()}),
                "affinenode" => Box::new(AffineNode { w, b, needed: weights.len()}),
                _ => panic!("Invalid node name {}", layer[0])
            };
            ls.push(t);
        }
        r.push(ls)
        
    }
    vec![]
}

#[test]
fn test_byte_vec() {
    create_byte_vec("model.prcs");
}