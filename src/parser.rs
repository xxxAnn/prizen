use std::{fs::{File, self}, path::Path, io::Read, io::Write};

use crate::{LinearNode, AffineNode, Node, Observation, mse, Model};

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

fn create_byte_vec<T: AsRef<Path>>(p: T, b: T) -> Model {
    let f = fs::read_to_string(p).expect("Couldn't read file");
    let fb = fs::read_to_string(b).expect("Couldn't read file");
    let lines = f.lines().into_iter().rev().collect::<Vec<&str>>();
    let mut meta = Vec::new();
    let mut lyers = Vec::new();
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
                    lyers.push(current_value);
                    current_value = Vec::new();
                } else {
                    current_value.push(l);
                }
                
            }
        }
    }
    meta.reverse();
    lyers.reverse();
    // Parsing layers
    let mut layers = Vec::new();
    for layer in lyers.iter() {
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
        layers.push(ls)
    }
    let obs: Observation = serde_json::from_str(&fb).unwrap();
    // Default values
    let mut inputs = 0;
    let mut alpha = 0.;
    let mut cst = Box::new(mse);
    for k in meta.iter() {
        let t = k.split_whitespace().collect::<Vec<&str>>();
        match t[0].to_lowercase().as_str() {
            "input" => inputs = t[1].parse::<usize>().unwrap(),
            "alpha" => alpha = t[1].parse::<f64>().unwrap(),
            "cst" => cst = match t[1].to_lowercase().as_str() {
                "mse" => Box::new(mse),
                _ => panic!("Invalid cost function name {}", t[1])
            },
            _ => panic!("Invalid meta value {}", t[0])
        }
    }

    Model {
        obs,
        cst,
        inputs,
        alpha,
        layers
    }
}

#[test]
fn test_byte_vec() {
    let mut mdl = create_byte_vec("model.prcs", "model.json");
    mdl.train(100000);
    let wbs = mdl.get_wb();
    println!("The Prizen's Guess: {:?}x + {:?}", wbs[0], wbs[1]);
}