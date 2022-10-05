use finitediff::FiniteDiff;

type Var = Vec<f64>;
type Column<T> = Vec<T>;
type PreviousColumn<T> = Vec<T>;
type Layer<T> = Vec<T>;
type SVar<'a> = &'a [f64];
// Represents any ANN node.
pub trait Node {
    fn get_value(&self, ord_vals: SVar, w: SVar, b: &f64) -> f64 {
        self.f(
            (0..ord_vals.len())
            .into_iter()
            .map(|i| ord_vals[i]*w[i])
            .sum::<f64>() + b
        )
    }
    
    fn f(&self, x: f64) -> f64;
}

pub struct LinearNode {
    w: Vec<f64>,
    b: f64
}

impl Node for LinearNode {
    fn f(&self, x: f64) -> f64 {
        x
    }
}

// Represents a real experimental value.
// ord_in: ordered inputs (in node order).
// ord_out: ordered outputs (in node order).
pub struct Observation {
    ord_in: Vec<Var>,
    ord_out: Vec<Var>
}

pub struct Model<'a, 'b> {
    obs: Observation,
    inputs: usize,
    layers: Vec<Vec<&'a dyn Node>>,
    alpha: f64,
    wb: WB,
    cst: &'b dyn Fn(Vec<Var>, Vec<Var>) -> Var
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

pub struct WB {
    // starts at layer 1
    w: Layer<Column<PreviousColumn<f64>>>,
    b: Layer<Column<f64>>,
    wbfr: [usize; 5]
}

pub struct Key {
    layer_count: usize, 
    column_lengths: Vec<usize>, // column_lengths.len() == layer_count
}

impl WB {
    pub fn flatten(&self) -> Vec<f64> {
        [self.w.concat().concat(), self.b.concat()].concat()
    }
    pub fn unflatten(k: Key, v: &[f64]) -> Self {
        todo!()
    }
}

impl Model<'_, '_> {
    pub fn calc(&self, v: Var) -> Var {
        let mut prev = v;
        let mut l = 1;
        let mut w;
        let mut b;
        for layer in self.layers.iter() {
            w = &self.wb.w[l];
            b = &self.wb.b[l];
            prev = (0..layer.len()).into_iter().map(|x| layer[x].get_value(&prev, &w[x], &b[x])).collect();
            l+=1;
        }
        prev
    }
    pub fn loss(&mut self, is: Vec<Var>, os: Vec<Var>) -> Var {
        let f = self.cst;

        f(is.into_iter().map(|x| self.calc(x, )).collect(), os)
    }
    pub fn update(&self) {
        
    }
}

/*
fn test() {
    let f = |x: &Vec<f64>| x[0].powi(2)+x[1].powi(2);
    
    println!("{:?}", vec![0.3, 0.4].forward_diff(&f));
    println!("{:?}", vec![0.3, 0.4].central_diff(&f));
}
*/