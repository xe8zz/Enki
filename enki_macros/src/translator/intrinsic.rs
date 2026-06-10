pub fn map_rust_method_to_slang(method_name: &str, base_expr: &str, args: &[String]) -> Option<String> {
    match method_name {
        "sin" => Some(format!("sin({})", base_expr)),
        "cos" => Some(format!("cos({})", base_expr)),
        "tan" => Some(format!("tan({})", base_expr)),
        "asin" => Some(format!("asin({})", base_expr)),
        "acos" => Some(format!("acos({})", base_expr)),
        "atan" => Some(format!("atan({})", base_expr)),
        "sinh" => Some(format!("sinh({})", base_expr)),
        "cosh" => Some(format!("cosh({})", base_expr)),
        "tanh" => Some(format!("tanh({})", base_expr)),
        "sqrt" => Some(format!("sqrt({})", base_expr)),
        "abs" => Some(format!("abs({})", base_expr)),
        "floor" => Some(format!("floor({})", base_expr)),
        "ceil" => Some(format!("ceil({})", base_expr)),
        "round" => Some(format!("round({})", base_expr)),
        "ln" => Some(format!("log({})", base_expr)),
        "log2" => Some(format!("log2({})", base_expr)),
        "log10" => Some(format!("log10({})", base_expr)),
        "exp" => Some(format!("exp({})", base_expr)),
        "exp2" => Some(format!("exp2({})", base_expr)),

        "powf" | "powi" => {
            if args.len() == 1 {
                Some(format!("pow({}, {})", base_expr, args[0]))
            } else {
                None
            }
        }
        "min" => {
            if args.len() == 1 {
                Some(format!("min({}, {})", base_expr, args[0]))
            } else {
                None
            }
        }
        "max" => {
            if args.len() == 1 {
                Some(format!("max({}, {})", base_expr, args[0]))
            } else {
                None
            }
        }
        "clamp" => {
            if args.len() == 2 {
                Some(format!("clamp({}, {}, {})", base_expr, args[0], args[1]))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn map_rust_function_to_slang(func_name: &str, args: &[String]) -> Option<String> {
    match func_name {
        "sin" | "cos" | "tan" | "sqrt" | "abs" | "floor" | "ceil" | "round" | "exp" | "log" | "log2" | "log10" => {
            if args.len() == 1 {
                Some(format!("{}({})", func_name, args[0]))
            } else {
                None
            }
        }
        "pow" | "min" | "max" | "step" => {
            if args.len() == 2 {
                Some(format!("{}({}, {})", func_name, args[0], args[1]))
            } else {
                None
            }
        }
        "clamp" | "mix" | "lerp" => {
            if args.len() == 3 {
                Some(format!("{}({}, {}, {})", func_name, args[0], args[1], args[2]))
            } else {
                None
            }
        }
        "dot" => {
            if args.len() == 2 {
                Some(format!("dot({}, {})", args[0], args[1]))
            } else {
                None
            }
        }
        "cross" => {
            if args.len() == 2 {
                Some(format!("cross({}, {})", args[0], args[1]))
            } else {
                None
            }
        }
        "normalize" => {
            if args.len() == 1 {
                Some(format!("normalize({})", args[0]))
            } else {
                None
            }
        }
        "length" => {
            if args.len() == 1 {
                Some(format!("length({})", args[0]))
            } else {
                None
            }
        }
        "distance" => {
            if args.len() == 2 {
                Some(format!("distance({}, {})", args[0], args[1]))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn map_rust_constant_to_slang(const_path: &str) -> Option<String> {
    if const_path.contains("PI") {
        Some("3.14159265358979323846".to_string())
    } else if const_path.contains("E") {
        Some("2.7182818284590452354".to_string())
    } else if const_path.contains("FRAC_1_PI") {
        Some("0.31830988618379067154".to_string())
    } else if const_path.contains("TAU") {
        Some("6.28318530717958647692".to_string())
    } else {
        None
    }
}