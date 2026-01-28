//! Pretty printing of tensors
//!
//! adapted from [candle](https://github.com/huggingface/candle)
//!
use crate::{FloatCategory, FloatDType, IntCategory, IntDType, NumCategory, NumDType, Result, Tensor, WithDType};

// =================================================================================== //
//                      Debug 
// =================================================================================== //

impl<T: WithDType> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor[")?;
        match self.dims() {
            [] => {
                if let Ok(v) = self.to_scalar() {
                    write!(f, "{v}")?
                }
            }
            [s] if *s < 10 => {
                for (i, v) in self.to_vec().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
            }
            dims => {
                write!(f, "dims ")?;
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{d}")?;
                }
            }
        }
        write!(f, "; {}]", self.dtype())
    }
}

// =================================================================================== //
//                  Options for Tensor pretty printing
// =================================================================================== //

#[derive(Debug, Clone)]
pub struct PrinterOptions {
    pub precision: usize,
    pub threshold: usize,
    pub edge_items: usize,
    pub line_width: usize,
    pub sci_mode: Option<bool>,
}

static PRINT_OPTS: std::sync::Mutex<PrinterOptions> =
    std::sync::Mutex::new(PrinterOptions::const_default());

impl PrinterOptions {
    // We cannot use the default trait as it's not const.
    const fn const_default() -> Self {
        Self {
            precision: 4,
            threshold: 1000,
            edge_items: 3,
            line_width: 80,
            sci_mode: None,
        }
    }
}

pub fn print_options() -> &'static std::sync::Mutex<PrinterOptions> {
    &PRINT_OPTS
}

pub fn set_print_options(options: PrinterOptions) {
    *PRINT_OPTS.lock().unwrap() = options
}

pub fn set_print_options_default() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions::const_default()
}

pub fn set_print_options_short() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 2,
        threshold: 1000,
        edge_items: 2,
        line_width: 80,
        sci_mode: None,
    }
}

pub fn set_print_options_full() {
    *PRINT_OPTS.lock().unwrap() = PrinterOptions {
        precision: 4,
        threshold: usize::MAX,
        edge_items: 3,
        line_width: 80,
        sci_mode: None,
    }
}

pub fn set_line_width(line_width: usize) {
    PRINT_OPTS.lock().unwrap().line_width = line_width
}

pub fn set_precision(precision: usize) {
    PRINT_OPTS.lock().unwrap().precision = precision
}

pub fn set_edge_items(edge_items: usize) {
    PRINT_OPTS.lock().unwrap().edge_items = edge_items
}

pub fn set_threshold(threshold: usize) {
    PRINT_OPTS.lock().unwrap().threshold = threshold
}

pub fn set_sci_mode(sci_mode: Option<bool>) {
    PRINT_OPTS.lock().unwrap().sci_mode = sci_mode
}

struct FmtSize {
    current_size: usize,
}

impl FmtSize {
    fn new() -> Self {
        Self { current_size: 0 }
    }

    fn final_size(self) -> usize {
        self.current_size
    }
}

impl std::fmt::Write for FmtSize {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.current_size += s.len();
        Ok(())
    }
}

trait TensorFormatter {
    type Elem: WithDType;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result;

    fn max_width(&self, to_display: &Tensor<Self::Elem>) -> usize {
        let mut max_width = 1;
        if let Ok(t) = to_display.flatten_all() {
            let vs = t.to_vec();
            for &v in vs.iter() {
                let mut fmt_size = FmtSize::new();
                let _res = self.fmt(v, 1, &mut fmt_size);
                max_width = usize::max(max_width, fmt_size.final_size())
            }
        }
        max_width
    }

    fn write_newline_indent(i: usize, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f)?;
        for _ in 0..i {
            write!(f, " ")?
        }
        Ok(())
    }

    fn fmt_tensor(
        &self,
        t: &Tensor<Self::Elem>,
        indent: usize,
        max_w: usize,
        summarize: bool,
        po: &PrinterOptions,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let dims = t.dims();
        let edge_items = po.edge_items;
        write!(f, "[")?;
        match dims {
            [] => {
                if let Ok(v) = t.to_scalar() {
                    self.fmt(v, max_w, f)?
                }
            }
            [v] if summarize && *v > 2 * edge_items => {
                if let Ok(t) = t
                    .narrow(0, 0, edge_items)
                {
                    for v in t.to_vec().into_iter() {
                        self.fmt(v, max_w, f)?;
                        write!(f, ", ")?;
                    }
                }
                write!(f, "...")?;
                if let Ok(t) = t
                    .narrow(0, v - edge_items, edge_items)
                {
                    for v in t.to_vec().into_iter() {
                        write!(f, ", ")?;
                        self.fmt(v, max_w, f)?;
                    }
                }
            }
            [_] => {
                let elements_per_line = usize::max(1, po.line_width / (max_w + 2));
                let vs = t.to_vec();
                for (i, v) in vs.into_iter().enumerate() {
                    if i > 0 {
                        if i % elements_per_line == 0 {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        } else {
                            write!(f, ", ")?;
                        }
                    }
                    self.fmt(v, max_w, f)?
                }
            }
            _ => {
                if summarize && dims[0] > 2 * edge_items {
                    for i in 0..edge_items {
                        match t.get(i) {
                            Ok(t) => self.fmt_tensor(&t, indent + 1, max_w, summarize, po, f)?,
                            Err(e) => write!(f, "{e:?}")?,
                        }
                        write!(f, ",")?;
                        Self::write_newline_indent(indent, f)?
                    }
                    write!(f, "...")?;
                    Self::write_newline_indent(indent, f)?;
                    for i in dims[0] - edge_items..dims[0] {
                        match t.get(i) {
                            Ok(t) => self.fmt_tensor(&t, indent + 1, max_w, summarize, po, f)?,
                            Err(e) => write!(f, "{e:?}")?,
                        }
                        if i + 1 != dims[0] {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        }
                    }
                } else {
                    for i in 0..dims[0] {
                        match t.get(i) {
                            Ok(t) => self.fmt_tensor(&t, indent + 1, max_w, summarize, po, f)?,
                            Err(e) => write!(f, "{e:?}")?,
                        }
                        if i + 1 != dims[0] {
                            write!(f, ",")?;
                            Self::write_newline_indent(indent, f)?
                        }
                    }
                }
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

struct FloatFormatter<S: FloatDType> {
    int_mode: bool,
    sci_mode: bool,
    precision: usize,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: FloatDType> FloatFormatter<S> {
    fn new(t: &Tensor<S>, po: &PrinterOptions) -> Result<Self> {
        let mut int_mode = true;
        let mut sci_mode = false;

        // Rather than containing all values, this should only include
        // values that end up being displayed according to [threshold].
        let values = t
            .flatten_all()?
            .to_vec()
            .into_iter()
            .filter(|v: &S| v.is_finite() && !v.is_zero())
            .collect::<Vec<_>>();
        if !values.is_empty() {
            let mut nonzero_finite_min = S::MAX_VALUE;
            let mut nonzero_finite_max = S::MIN_VALUE;
            for &v in values.iter() {
                let v = v.abs();
                if v < nonzero_finite_min {
                    nonzero_finite_min = v
                }
                if v > nonzero_finite_max {
                    nonzero_finite_max = v
                }
            }

            for &value in values.iter() {
                if value.ceil() != value {
                    int_mode = false;
                    break;
                }
            }
            if let Some(v1) = S::from(1000.) {
                if let Some(v2) = S::from(1e8) {
                    if let Some(v3) = S::from(1e-4) {
                        sci_mode = nonzero_finite_max / nonzero_finite_min > v1
                            || nonzero_finite_max > v2
                            || nonzero_finite_min < v3
                    }
                }
            }
        }

        match po.sci_mode {
            None => {}
            Some(v) => sci_mode = v,
        }
        Ok(Self {
            int_mode,
            sci_mode,
            precision: po.precision,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<S: FloatDType> TensorFormatter for FloatFormatter<S> {
    type Elem = S;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result {
        if self.sci_mode {
            write!(
                f,
                "{v:width$.prec$e}",
                v = v,
                width = max_w,
                prec = self.precision
            )
        } else if self.int_mode {
            if v.is_finite() {
                write!(f, "{v:width$.0}.", v = v, width = max_w - 1)
            } else {
                write!(f, "{v:max_w$.0}")
            }
        } else {
            write!(
                f,
                "{v:width$.prec$}",
                v = v,
                width = max_w,
                prec = self.precision
            )
        }
    }
}

struct IntFormatter<S: WithDType> {
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WithDType> IntFormatter<S> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S: WithDType> TensorFormatter for IntFormatter<S> {
    type Elem = S;

    fn fmt<T: std::fmt::Write>(&self, v: Self::Elem, max_w: usize, f: &mut T) -> std::fmt::Result {
        write!(f, "{v:max_w$}")
    }
}

fn get_summarized_data<T: WithDType>(t: &Tensor<T>, edge_items: usize) -> Result<Tensor<T>> {
    let dims = t.dims();
    if dims.is_empty() {
        Ok(t.clone())
    } else if dims.len() == 1 {
        if dims[0] > 2 * edge_items {
            Tensor::cat(
                &[
                    t.narrow(0, 0, edge_items)?,
                    t.narrow(0, dims[0] - edge_items, edge_items)?,
                ],
                0,
            )
        } else {
            Ok(t.clone())
        }
    } else if dims[0] > 2 * edge_items {
        let mut vs: Vec<_> = (0..edge_items)
            .map(|i| get_summarized_data(&t.get(i)?, edge_items))
            .collect::<Result<Vec<_>>>()?;
        for i in (dims[0] - edge_items)..dims[0] {
            vs.push(get_summarized_data(&t.get(i)?, edge_items)?)
        }
        Tensor::cat(&vs, 0)
    } else {
        let vs: Vec<_> = (0..dims[0])
            .map(|i| get_summarized_data(&t.get(i)?, edge_items))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&vs, 0)
    }
}

trait Display<T: WithDType> {
    fn fmt_display(tensor: &Tensor<T>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let po: std::sync::MutexGuard<'_, PrinterOptions> = PRINT_OPTS.lock().unwrap();
        let summarize = tensor.element_count() > po.threshold;
        let to_display = if summarize {
            match get_summarized_data(tensor, po.edge_items) {
                Ok(v) => v,
                Err(err) => return write!(f, "{err:?}"),
            }
        } else {
            tensor.clone()
        };

        Self::do_fmt_display(tensor, to_display, f, summarize, po)?;

        write!(
            f,
            "Tensor[{:?}, {}]",
            tensor.dims(),
            tensor.dtype(),
        )
    }

    fn do_fmt_display(
        tensor: &Tensor<T>, 
        to_display: Tensor<T>, 
        f: &mut std::fmt::Formatter,
        summarize: bool,
        po: std::sync::MutexGuard<'_, PrinterOptions>
    ) -> std::fmt::Result {
        let tf = IntFormatter::new();
        let max_w = tf.max_width(&to_display);
        tf.fmt_tensor(tensor, 1, max_w, summarize, &po, f)?;
        writeln!(f)
    }
}

struct DisplayOp;

impl<T: NumDType> Display<T> for DisplayOp 
where 
    Self: DisplayCategory<T>,
{
    fn do_fmt_display(
        tensor: &Tensor<T>, 
        to_display: Tensor<T>, 
        f: &mut std::fmt::Formatter,
        summarize: bool,
        po: std::sync::MutexGuard<'_, PrinterOptions>
    ) -> std::fmt::Result {
        DisplayOp::do_fmt_display_category(tensor, to_display, f, summarize, po)
    }
}

trait DisplayCategory<T: NumDType, C: NumCategory = <T as NumDType>::Category> {
    fn do_fmt_display_category(
        tensor: &Tensor<T>, 
        to_display: Tensor<T>, 
        f: &mut std::fmt::Formatter,
        summarize: bool,
        po: std::sync::MutexGuard<'_, PrinterOptions>
    ) -> std::fmt::Result;
}

impl<T: IntDType> DisplayCategory<T, IntCategory> for DisplayOp {
    fn do_fmt_display_category(
        tensor: &Tensor<T>, 
        to_display: Tensor<T>, 
        f: &mut std::fmt::Formatter,
        summarize: bool,
        po: std::sync::MutexGuard<'_, PrinterOptions>
    ) -> std::fmt::Result {
        let tf: IntFormatter<T> = IntFormatter::new();
        let max_w = tf.max_width(&to_display);
        tf.fmt_tensor(tensor, 1, max_w, summarize, &po, f)?;
        writeln!(f)
    }
} 

impl<T: FloatDType> DisplayCategory<T, FloatCategory> for DisplayOp {
    fn do_fmt_display_category(
        tensor: &Tensor<T>, 
        to_display: Tensor<T>, 
        f: &mut std::fmt::Formatter,
        summarize: bool,
        po: std::sync::MutexGuard<'_, PrinterOptions>
    ) -> std::fmt::Result {
        if let Ok(tf) = FloatFormatter::new(&to_display, &po) {
            let max_w = tf.max_width(&to_display);
            tf.fmt_tensor(tensor, 1, max_w, summarize, &po, f)?;
            writeln!(f)?;
        }
        Ok(())
    }
} 

impl Display<bool> for DisplayOp {
    
}

impl<T: WithDType> std::fmt::Display for Tensor<T> 
where 
    DisplayOp: Display<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        DisplayOp::fmt_display(self, f)
    }
}