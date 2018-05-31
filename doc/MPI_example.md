# Using MPI

In theory, our example can easily be ported to run on a cluster. The `emcee` docs give an example of how to get this working [here](http://dfm.io/emcee/current/user/advanced/#using-mpi-to-distribute-the-computations). The file `MPI_fit_example.py` gives a simple implementation of this method which I've found to work well enough for my purposes. 

MPI works by having each computing node run the same file, with one 'master' process distributing tasks to them. I've found that the key to getting MPI to work nicely with PyStaff is to have every node run `fit.set_up_fit()`, so that each one has a 'local' copy of `fit.fit_settings`. Each node also has a 'local' implementation of the `lnprob` function. Whilst this isn't very memory efficient, is does mean that we don't have to pickle the `fit_settings` dictionary at every step- if we do that, I've found it leads to a slow down by factors of 30-50 (see the discussion on the `emcee` github page [here](shttps://github.com/dfm/emcee/blob/master/docs/tutorials/parallel.rst)

A nicer way to do this would be to load everything on the master process,  broadcast the necessary bits to each node and then do the fitting. But I've never had the time to code this up properly, especially as the above method works as well as I need it to. 
