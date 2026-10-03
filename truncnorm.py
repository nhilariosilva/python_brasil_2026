import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

@tf.custom_gradient
def log_pdf_dmu(y, mu, sigma):
    # Standardize the variables
    z = (y - mu) / sigma
    t = mu / sigma
    
    # Standard Normal Distribution
    dist = tfd.Normal(loc=0.0, scale=1.0)
    
    # Inverse Mills Ratio h(t) = phi(t) / Phi(t)
    # Computed strictly in log-space for numerical stability
    h = tf.math.exp(dist.log_prob(t) - dist.log_cdf(t))
    
    # First derivative of log-likelihood w.r.t mu
    first_derivative_mu = (z - h) / sigma

    def custom_second_derivative(upstream_grad):
        # Derivative of the Inverse Mills Ratio: h'(t) = -h(t) * (h(t) + t)
        h_prime = -h * (h + t)
        
        d2log_pdf_y_mu2 = (-1.0 - h_prime) / (sigma ** 2.0)
        d2log_pdf_y_musigma = (h - 2.0 * z + t * h_prime) / (sigma ** 2.0)

        jac_mu2 = upstream_grad * d2log_pdf_y_mu2
        jac_musigma = upstream_grad * d2log_pdf_y_musigma
        
        # Return derivatives w.r.t (y, mu, sigma). y is None.
        return None, jac_mu2, jac_musigma
    
    return first_derivative_mu, custom_second_derivative

@tf.custom_gradient
def log_pdf_dsigma(y, mu, sigma):
    z = (y - mu) / sigma
    t = mu / sigma
    
    dist = tfd.Normal(loc=0.0, scale=1.0)
    h = tf.math.exp(dist.log_prob(t) - dist.log_cdf(t))
    
    # First derivative of log-likelihood w.r.t sigma
    first_derivative_sigma = (z**2.0 - 1.0 + h * t) / sigma

    def custom_second_derivative(upstream_grad):
        h_prime = -h * (h + t)
        
        d2log_pdf_y_musigma = (h - 2.0 * z + t * h_prime) / (sigma ** 2.0)
        d2log_pdf_y_sigma2 = (1.0 - 3.0 * z**2.0 - 2.0 * h * t - (t**2.0) * h_prime) / (sigma ** 2.0)

        jac_musigma = upstream_grad * d2log_pdf_y_musigma
        jac_sigma2 = upstream_grad * d2log_pdf_y_sigma2
        
        # Return derivatives w.r.t (y, mu, sigma). y is None.
        return None, jac_musigma, jac_sigma2
    
    return first_derivative_sigma, custom_second_derivative

@tf.custom_gradient
def log_pdf(y, mu, sigma):
    # Enforce float32
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)

    z = (y - mu) / sigma
    t = mu / sigma
    
    dist = tfd.Normal(loc=0.0, scale=1.0)
    
    # Log-Likelihood of Truncated Normal bounded at 0
    # log(phi(z)) - log(sigma) - log(Phi(t))
    log_pdf_y = dist.log_prob(z) - tf.math.log(sigma) - dist.log_cdf(t)

    def custom_derivative(upstream_grad):
        # Route directly to our analytical first and second derivatives
        dlog_pdf_y_mu = log_pdf_dmu(y, mu, sigma)
        dlog_pdf_y_sigma = log_pdf_dsigma(y, mu, sigma)
        
        grad_mu = upstream_grad * dlog_pdf_y_mu
        grad_sigma = upstream_grad * dlog_pdf_y_sigma
        
        return None, grad_mu, grad_sigma
              
    return log_pdf_y, custom_derivative

@tf.function(reduce_retracing=True)
def pdf(y, mu, sigma):
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)

    z = (y - mu) / sigma
    t = mu / sigma
    
    dist = tfd.Normal(loc=0.0, scale=1.0)
    
    # pdf(y) = phi(z) / (sigma * Phi(t))
    return dist.prob(z) / (sigma * dist.cdf(t))

@tf.function(reduce_retracing=True)
def cdf(y, mu, sigma):
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)

    z = (y - mu) / sigma
    t = mu / sigma
    
    dist = tfd.Normal(loc=0.0, scale=1.0)
    
    # F(y) = [Phi(z) - Phi(-t)] / Phi(t)
    return (dist.cdf(z) - dist.cdf(-t)) / dist.cdf(t)

@tf.function(reduce_retracing=True)
def ppf(q, mu, sigma):
    q = tf.cast(q, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)
    
    t = mu / sigma
    dist = tfd.Normal(loc=0.0, scale=1.0)
    
    # Inverse CDF logic for Truncated Normal
    # P(Y <= y) = q  =>  Phi(z) = q * Phi(t) + Phi(-t)
    target_prob = q * dist.cdf(t) + dist.cdf(-t)
    
    # Clip prob safely between 0 and 1 to prevent inf quantiles
    target_prob = tf.clip_by_value(target_prob, 1e-7, 1.0 - 1e-7)
    
    z_q = dist.quantile(target_prob)
    y_q = mu + sigma * z_q
    
    return y_q