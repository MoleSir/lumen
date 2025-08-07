#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Range {
    Index{ index: usize, used: bool },
    Range{ start: usize, end: Option<usize>, step: usize },
}

impl Range {
    pub fn range(start: usize, end: usize, step: usize) -> Self {
        Self::Range { start, end: Some(end), step }
    }

    pub fn range_no_end(start: usize, step: usize) -> Self {
        Self::Range { start, end: None, step }
    }

    pub fn index(index: usize) -> Self {
        Self::Index { index, used: false }
    }

    pub fn is_range(&self) -> bool {
        if let Self::Range { start:_, end:_, step:_ } = self {
            true
        } else {
            false
        }
    }
}

impl Iterator for Range {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Index { index, used } => {
                if *used == false {
                    *used = true;
                    Some(*index)
                } else {
                    None
                }
            }
            Self::Range { start, end, step } => {
                match end {
                    Some(end) => {
                        if start < end {
                            let value = *start;
                            *start += *step;
                            Some(value)
                        } else {
                            None
                        }
                    }
                    None => {
                        let value = *start;
                        *start += *step;
                        Some(value)
                    }
                }
            }
        }
    }
}

#[macro_export]
macro_rules! rng {
    // rng!(start:end)
    ($start:tt : $end:tt) => {
        Range::range($start as usize, $end as usize, 1)
    };
    // rng!(start:end:step)
    ($start:tt : $end:tt : $step:tt) => {
        Range::range($start as usize, $end as usize, $step as usize)
    };
    // rng!(start:)
    ($start:tt :) => {
        Range::range_no_end($start as usize, 1)
    };
    // rng!(start::step)
    ($start:tt :: $step:tt) => {
        Range::range_no_end($start as usize, $step as usize)
    };
    // rng!(:$end)
    (: $end:tt) => {
        Range::range(0, $end as usize, 1)
    };
    // rng!(:$end:$step)
    (: $end:tt : $step:tt) => {
        Range::range(0, $end as usize, $step as usize)
    };
    // rng!(::$step)
    (:: $step:tt) => {
        Range::range_no_end(0, $step as usize)
    };
    // rng!(:)
    (:) => {
        Range::range_no_end(0, 1)
    };
    // rng!(:)
    ($index:tt) => {
        Range::index($index as usize)
    };
}

// rngs!(:, 123:)
#[macro_export]
macro_rules! rngs {
    ($x:tt) => {
        &[rng!($x)]
    };

    (($x:tt), $(($rest:tt)),*) => {
        &[rng!($x), $(rng!($rest)),*]
    };
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use super::*;
    
    #[test]
    fn test_macro() {
        let t = (0..12usize);
        let t = (2usize..);
        assert_eq!(rng!(1:10), Range::Range {start:1, end: Some(10), step:1});

        assert!(
            rng!(1:20).zip((1..20))
                .all(|(a, b)| a == b)
        );
    
        assert!(
            rng!(1:13:3).zip((1..13).step_by(3))
                .all(|(a, b)| a == b)
        );
    
        assert!(
            rng!(1:).zip((1..).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(1::2).zip((1..).step_by(2).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(:20).zip((0..20usize))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(:20:5).zip((0..20usize).step_by(5))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(::2).zip((0..).step_by(2).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(:).zip((0..).take(100))
                .all(|(a, b)| a == b)
        );

        assert!(
            rng!(1).zip((1..)).take(1)
                .all(|(a, b)| a == b)
        );

        assert_eq!(
            rngs!(1),
            &[rng!(1)],
        );
    
        assert_eq!(
            rngs!((1), (:), (:)),
            &[rng!(1), rng!(:), rng!(:)],
        );
    }
}

