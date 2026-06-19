
pub trait Model {
    type Configuration: ModelConfiguration;

    fn fit(x, y)
}

pub trait ModelConfiguration {

}

