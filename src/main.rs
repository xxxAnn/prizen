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
    b: f64,
    id: usize
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

#[derive(Debug)]
pub struct WB {
    // starts at layer 1
    w: Layer<Column<PreviousColumn<f64>>>,
    b: Layer<Column<f64>>,
}
// [[[a; 1]; 1]; 1]
pub struct Key {
    layer_count: usize, 
    column_lengths: Vec<usize>, // column_lengths.len() == layer_count+1 since the input layer isn't counted
}

impl WB {
    pub fn flatten(&self) -> Vec<f64> {
        [self.w.concat().concat(), self.b.concat()].concat()
    }
    pub fn unflatten(k: Key, v: &[f64]) -> Self {
        let mut total = 0;
        let mut n = 1;
        while n < k.column_lengths.len() {
            total+=k.column_lengths[n]*k.column_lengths[n-1];
            n+=1;
        }
        let w_stuff = &v[0..total];
        let b_stuff = &v[total..];
        let mut my_b = Vec::new();
        let mut my_w = Vec::new();
        let mut s = 0;
        let mut j: usize = 0;
        let mut v = k.column_lengths[0];
        for l in &k.column_lengths[1..] {
            println!("l: {}", l);
            my_b.push(Vec::from(&b_stuff[s..*l]));
            my_w.push(Vec::from(&w_stuff[j..(v*l)]));
            s = *l;
            v = *l;
            j = v*l;
        }
        // [[]]
        let mut f_w = Vec::new();
        let mut i = 1;
        for sub_w in my_w {
            println!("sub_w: {:?}", &sub_w);
            let mut sub_f_w = Vec::new();
            let mut s = 0;
            for _ in 0..k.column_lengths[i] {
                println!("sub_w l: {:?}", &(s..k.column_lengths[i-1]));
                sub_f_w.push(Vec::from(&sub_w[s..s+k.column_lengths[i-1]]));
                s += k.column_lengths[i-1];
            }
            i+=1;
            f_w.push(sub_f_w);
        }

        WB {
            w: f_w,
            b: my_b
        }
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

fn main() {
    let my_wb = WB {
        w: vec![vec![vec![4., 4.]]],
        b: vec![vec![4.]]
    };
    dbg!(&my_wb);
    dbg!(WB::unflatten(Key {
        layer_count: 1usize,
        column_lengths: vec![2, 1]
    }, &my_wb.flatten()));
    
}