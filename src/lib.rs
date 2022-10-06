use finitediff::FiniteDiff;

type Var = Vec<f64>;
type SVar<'a> = &'a [f64];

type Ribbon = Vec<f64>;

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

    fn take_w(&mut self, v: Vec<f64>) -> Vec<f64>;

    fn take_b(&mut self, v: Vec<f64>) -> Vec<f64>;

    fn bias(&self) -> f64;

    fn weight(&self, i: usize) -> f64;

    fn needed(&self) -> usize; // amount of weights needed
    
    fn f(&self, x: f64) -> f64;

    fn weights(&self) -> Vec<f64>;
}

#[derive(Clone)]
pub struct LinearNode {
    w: Vec<f64>,
    b: f64,
    needed: usize // number of weights needed
}

impl Node for LinearNode {
    fn f(&self, x: f64) -> f64 {
        x
    }

    fn needed(&self) -> usize {
        self.needed
    }

    fn take_b(&mut self, v: Vec<f64>) -> Vec<f64> {
        self.b = v[v.len()];
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
pub struct Observation {
    ord_in: Vec<Var>,
    ord_out: Vec<Var>
}

pub struct Model {
    obs: Observation,
    inputs: usize,
    layers: Vec<Vec<Box<dyn Node>>>,
    alpha: f64,
    cst: Box<dyn Fn(Vec<Var>, Vec<Var>) -> Var>
}

pub fn mse(a: Vec<Var>, b: Vec<Var>) -> Var {
    let x =(0..a.len()).map(|i| {
        sqdf(&a[i], &b[i])
    }).collect::<Vec<Var>>();

    let v = sum(&x);

    v.into_iter().map(|z| z/x.len() as f64).collect()
}

fn sqdf(a: SVar, b: SVar) -> Var {
    (0..a.len()).map(|i| {
        (a[i] - b[i]).powi(2)
    }).collect()
}

fn sum(a: &Vec<Vec<f64>>) -> Vec<f64> {
    let f = &mut (a[0].clone());
    for i in 1..f.len() {
        for j in 0..a[i].len() {
            f[j] += a[i][j];
        }
    }
    f.clone()
}

impl Model {
    pub fn calc(&self, v: Vec<f64>) -> Var {
        let mut prev = v;
        for layer in self.layers.iter() {
            prev = (0..layer.len()).into_iter().map(|x| layer[x].get_value(&prev)).collect();
        }
        prev
    }
    pub fn loss(&mut self, is: Vec<Var>, os: Vec<Var>) -> Var {
        let f = &self.cst;

        f(is.into_iter().map(|x| self.calc(x, )).collect(), os)
    }
    pub fn update_wb(&mut self, v: Vec<f64>) {
        let mut z = v.clone();
        for x in self.layers.iter_mut().rev() { for y in x {
            z = y.take_b(z)
        }}
        for x in self.layers.iter_mut().rev() { for y in x {
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
}