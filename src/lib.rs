use serde::{Deserialize, Serialize};
use serde_json::Result;
use bincode;

type Var = Vec<f64>;
type SVar<'a> = &'a [f64];

type Ribbon = Vec<f64>;

pub mod parser;

// Represents any ANN node.
pub trait Node {
    fn get_value(&self, ord_vals: SVar) -> f64 {
        self.f(
            (0..ord_vals.len())
            .into_iter()
            .map(|i| ord_vals[i]*self.weight(i))
            .sum::<f64>() + self.bias()
        )
    }
    // takes its own weights and biases from the ribbon

    fn take_w(&mut self, v: Ribbon) -> Ribbon;

    fn take_b(&mut self, v: Ribbon) -> Ribbon;

    fn bias(&self) -> f64;

    fn weight(&self, i: usize) -> f64;

    fn needed(&self) -> usize; // amount of weights needed
    
    fn f(&self, x: f64) -> f64;

    fn weights(&self) -> Vec<f64>;
}

#[derive(Clone, Debug)]
pub struct LinearNode {
    w: Vec<f64>,
    b: f64,
    needed: usize // number of weights needed
}

impl Node for LinearNode {
    fn f(&self, x: f64) -> f64 {
        x-self.bias()
    }

    fn needed(&self) -> usize {
        self.needed
    }

    fn take_b(&mut self, v: Vec<f64>) -> Vec<f64> {
        self.b = v[v.len()-1];
        v[0..v.len()-1].into()
    }
    
    fn take_w(&mut self, v: Vec<f64>) -> Vec<f64> {
        self.w = v[v.len()-self.needed()..v.len()].into();
        v[0..v.len()-self.needed()].into()
    }
    
    fn bias(&self) -> f64 {
        self.b
    }

    fn weight(&self, i: usize) -> f64 {
        self.w[i]
    }

    fn weights(&self) -> Vec<f64> {
        self.w.clone()
    }
}

#[derive(Clone, Debug)]
pub struct AffineNode {
    w: Vec<f64>,
    b: f64,
    needed: usize // number of weights needed
}

impl Node for AffineNode {
    fn f(&self, x: f64) -> f64 {
        x
    }

    fn needed(&self) -> usize {
        self.needed
    }

    fn take_b(&mut self, v: Vec<f64>) -> Vec<f64> {
        self.b = v[v.len()-1];
        v[0..v.len()-1].into()
    }
    
    fn take_w(&mut self, v: Vec<f64>) -> Vec<f64> {
        self.w = v[v.len()-self.needed()..v.len()].into();
        v[0..v.len()-self.needed()].into()
    }
    
    fn bias(&self) -> f64 {
        self.b
    }

    fn weight(&self, i: usize) -> f64 {
        self.w[i]
    }

    fn weights(&self) -> Vec<f64> {
        self.w.clone()
    }
}


// Represents a real experimental value.
// ord_in: ordered inputs (in node order).
// ord_out: ordered outputs (in node order).
#[derive(Serialize, Deserialize)]
pub struct Observation {
    ord_in: Vec<Var>,
    ord_out: Vec<Var>
}

pub struct Model {
    pub obs: Observation,
    pub inputs: usize,
    pub layers: Vec<Vec<Box<dyn Node>>>,
    pub alpha: f64,
    pub cst: Box<dyn Fn(Vec<Vec<f64>>, Vec<Vec<f64>>) -> f64>
}

pub fn mse(a: Vec<Var>, b: Vec<Var>) -> f64 {
    let f = |c: &[f64], d: &[f64]| -> f64 {
        let x: f64 = (0..c.len()).map(|i| {
            (c[i]-d[i]).powi(2)
        }).collect::<Vec<f64>>().iter().sum();

        x/(c.len() as f64)
    };
    let x: f64 =(0..a.len()).map(|i| {
        (f(&a[i], &b[i])).powi(2)
    }).collect::<Vec<f64>>().iter().sum();

    x/(a.len() as f64)
}


impl Model {
    pub fn calc(&self, v: &[f64]) -> Vec<f64> {
        let mut prev = v.to_vec();
        for layer in self.layers.iter() {
            prev = (0..layer.len()).into_iter().map(|x| layer[x].get_value(&prev)).collect();
        }
        prev
    }
    pub fn loss(&self, is: &[Var], os: &[Var]) -> f64 {
        let f = &self.cst;

        f(is.into_iter().map(|x| self.calc(&x)).collect(), os.to_vec())
    }
    pub fn update_wb(&mut self, v: &[f64]) {
        let mut z = v.to_vec();
        for x in self.layers.iter_mut().rev() { for y in x.iter_mut().rev() {
            z = y.take_b(z)
        }}
        for x in self.layers.iter_mut().rev() { for y in x.iter_mut().rev() {
            z = y.take_w(z)
        }}
    }
    pub fn get_wb(&self) -> Vec<f64> {
        let mut wres = Vec::new();
        let mut bres = Vec::new();
        for x in &self.layers { for y in x {
            bres.push(y.bias());
            wres.push(y.weights());
        }}
        [wres.concat(), bres].concat()
    }
    pub fn update_wbs(&mut self) {
        let mut r = Vec::new();
        let wbs = self.get_wb();
        let h = 1e-12; // tolerance
        for i in 0..wbs.len() {
            let mut t = wbs.clone();
            t[i] = t[i]+h;
            self.update_wb(&t);
            let a = self.loss(&self.obs.ord_in,& self.obs.ord_out);
            //println!("Loss: {:?}", &a);
            t[i] = t[i]-h;
            self.update_wb(&t);
            let b = self.loss(&self.obs.ord_in,&self.obs.ord_out);
            r.push(wbs[i]-(self.alpha*((a-b)/(2.*h))));
        }
        self.update_wb(&r);
    }
    pub fn train(&mut self, iter: usize) {
        let mut n = 0;
        while n < iter {
            self.update_wbs();
            if n % 1000 == 0 {
                println!("{:?}x + {:?}", self.get_wb()[0], self.get_wb()[1]);
            }
            n+=1;
        }
    }
}

#[test]
fn test() {
    let mut mdl = Model {
        obs: Observation { ord_in: vec![vec![10.], vec![20.], vec![25.], vec![30.]], ord_out: vec![vec![22.5], vec![46.5], vec![61.1], vec![70.]] },
        inputs: 1,
        layers: vec![vec![Box::new(LinearNode {
            w: vec![1.],
            b: 0.,
            needed: 1
        })]],
        alpha: 0.0001,
        cst: Box::new(mse)
    };
    mdl.train(100000); // train 100,000 iterations
    let wbs = mdl.get_wb();
    println!("The Prizen's Guess: {:?}x + {:?}", wbs[0], wbs[1]);
}