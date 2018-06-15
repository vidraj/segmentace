#!/usr/bin/env perl

use utf8;
use strict;
use warnings;


# I assume that all IO will be in UTF-8. Make sure to recode all input files!
binmode STDIN, "utf8";
binmode STDOUT, "utf8";
binmode STDERR, "utf8";

use List::Util qw(sum0);


# Load the words from a file into an array. The array is prepended with two dummy symbols to preserve the first two trigrams.
# I use newlines for marking the two positions before the beginning of the text itself. A newline cannot be in the text, so this is safe.
sub read_data_from_file {
	my $file_name = shift;
	
	# Read the data …
	open my $file, '<:encoding(UTF-8)', $file_name
		or die "Can't open file '$file_name': $!\n";

	# … into an array in a fashion similar to the training data
	my @data = ("\n\n", "\n");
	while (<$file>) {
		chomp;
		push @data, $_;
	}

	close $file
		or die "IO error while closing file '$file_name': $!\n";
	
	return @data;
}

# Filter an array argument, erasing duplicate values, so that every value in @_ is returned exactly once.
# Used for measuring vocabulary size.
sub uniq {
	my %seen;
	return grep { !$seen{$_}++ } @_;
}

# Calculate binary logarithm of the single numerical argument
sub log2 {
	return log(shift) / log(2);
}





# Check that the commandline parameters are right.
if (@ARGV != 3) {
	die "Usage:\n\tcat test-data | $0 train-data-filename heldout-data-filename test-data-filename\n\nError: Expected three filenames – of the training, heldout and testing data – as arguments.\n";
}

# The first argument is used as the training data file
my $train_file_name = $ARGV[0];
# We use this file for the smoothing EM algorithm
my $heldout_file_name = $ARGV[1];
#  … and this one for testing.
my $test_file_name = $ARGV[2];

# An epsilon value used for stopping lambda-recalculating loop during Linear Interpolation parameter training
# If any of the old and new lambdas differ more than the threshold, the iteration is continued.
my $stopping_threshold = 0.0001;






# The training text file, parsed on line-endings to form an array of words in the original order.
my @words = read_data_from_file($train_file_name);

# Load the heldout data into an array as well.
my @heldout = read_data_from_file($heldout_file_name);




# Estimate the language vocabulary size by using the computed vocab. size of the test + heldout data.
my $vocabulary_size = uniq(@words, @heldout) - 2; # Minus 2 because of the start-of-text markers


# Counts of various n-grams in the training data
my (%trigram_cts, %bigram_cts, %unigram_cts);


# Takes three arguments – an array of words to process and three hash references.
# Will calculate the {uni,bi,tri}gram counts from the array and store them in the appropriate hashes.
# Returns total number of trigrams processed.
# Beware of sideeffects!
sub calculate_ngram_counts {
	my ($words_ref, $unigram_cts_ref, $bigram_cts_ref, $trigram_cts_ref) = @_;
	my @words = @$words_ref;
	
	# I count the trigrams from data and then compute bigrams and unigrams by summation.
	# I ignore the first two words – they are used for history only, not for counting.
	my $trigrams_total;
	for (my $i = 2; $i < @words; $i++) {
		$trigram_cts_ref->{$words[$i-2]}->{$words[$i-1]}->{$words[$i]}++;
		$trigrams_total++;
	}

	# Compute lesser-grams by summation.
	for my $w2 (keys %$trigram_cts_ref) {
		for my $w1 (keys %{$trigram_cts_ref->{$w2}}) {
			for my $w (keys %{$trigram_cts_ref->{$w2}->{$w1}}) {
				$bigram_cts_ref->{$w1}->{$w} += $trigram_cts_ref->{$w2}->{$w1}->{$w};
				$unigram_cts_ref->{$w}       += $trigram_cts_ref->{$w2}->{$w1}->{$w};
			}
		}
	}
	
	return $trigrams_total;
}


my $trigrams_total = calculate_ngram_counts(\@words, \%unigram_cts, \%bigram_cts, \%trigram_cts);





## Subroutines for probability calculations.
# Be aware that the arguments contain history in right-to-left fashion;
#  i.e. oldest word is in the rightmost argument.

sub p0 {
	return 1/$vocabulary_size;
}

# p₁(w)
sub p1 {
	my ($w) = @_;
	
	# Prevent warning that $w is undefined.
	return 0 if (!defined $unigram_cts{$w});
	
	return $unigram_cts{$w} / $trigrams_total;
}

# p₂(w | w1)
sub p2 {
	my ($w, $w1) = @_;
	
	# Prevent division by zero.
	return 0 if (!defined($unigram_cts{$w1}) or !defined($bigram_cts{$w1}->{$w}) or $unigram_cts{$w1} == 0);
	
	return $bigram_cts{$w1}->{$w} / $unigram_cts{$w1};
}

# p₃(w | w2, w1)
sub p3 {
	my ($w, $w1, $w2) = @_;
	
	# Prevent division by zero.
	return 0 if (!defined($bigram_cts{$w2}->{$w1}) or !defined($trigram_cts{$w2}->{$w1}->{$w}) or $bigram_cts{$w2}->{$w1} == 0);
	
	return $trigram_cts{$w2}->{$w1}->{$w} / $bigram_cts{$w2}->{$w1};
}

#####




# Select a starting set of lambdas. Any nonzero will do.
my @lambdas = (0.25, 0.25, 0.25, 0.25);

# References to the probability functions – for ease of use.
my @p = (\&p0, \&p1, \&p2, \&p3);

# Final – smoothed – probability.
sub ps {
	my ($lambdas_ref, $w, $w1, $w2) = @_;
	return sum0 map { $lambdas_ref->[$_] * $p[$_]->($w, $w1, $w2) } 0 .. (@$lambdas_ref - 1);
}


sub expected_count_of_lambdas {
	# Calculating counts for j-th lambda using the $text_ref as an array containing the words.
	# (Step 1 of the EM algorithm.)
	my ($j, $text_ref) = @_;
	my @text = @$text_ref;
	
	my $count = 0;
	for (my $i = 2; $i < @text; $i++) {
		my $smoothed_prob = ps(\@lambdas, $text[$i], $text[$i-1], $text[$i-2]);
		
		# Avoid division by zero.
		next if ($smoothed_prob == 0);
		
		$count += $lambdas[$j] * $p[$j]->($text[$i], $text[$i-1], $text[$i-2]) / $smoothed_prob;
	}
	
	return $count;
}

# A flag describing whether we should stop recalculating lambdas
my $converged = 0;


# Train the lambdas on the heldout data using the expectation-maximization algorithm
while (!$converged) {
# 	print "Looping! Lambdas are: " . join(", ", @lambdas) . "\n";
	
	# Calculate the expected lambda counts -- step 1 of the EM algorithm
	my @lambda_cts = map { expected_count_of_lambdas($_, \@heldout) } 0 .. $#lambdas;
	my $lambda_sum = sum0 @lambda_cts;
	
	# Update the lambdas by scaling them according to their counts -- step 2 of the EM algorithm
	my @new_lambdas = map { $_ / $lambda_sum } @lambda_cts;
	
# 	print "Lambda counts are: " . join(", ", @lambda_cts) . "\n";
	
	# Test whether all old-new lambda differences are below the threshold
	$converged = 1;
	for (my $i = 0; $i < $#lambdas; $i++) {
		$converged = 0 if (abs($lambdas[$i] - $new_lambdas[$i]) > $stopping_threshold); # No, there is a pair with a larger difference. Therefore, continue looping.
	}
	
	@lambdas = @new_lambdas;
}


# Delete the training and heldout corpora.
undef @words;
undef @heldout;

print STDERR "Lambdas for $heldout_file_name: " . join(", ", @lambdas) . "\n";






### Cross entropy

# Calculate cross entropy between the $test_data_ref words array and the "ps" function with parameters from $lambdas_ref
sub cross_entropy {
	my ($lambdas_ref, $test_file_name) = @_;
	
	# Read the data …
	open my $test_file, '<:encoding(UTF-8)', $test_file_name
		or die "Can't open file '$test_file_name': $!\n";
	
	
	my $test_trigrams_total = 0;
	my $centropy = 0.0;
	

	# … into an array in a fashion similar to the training data
	my @test_data = ("", "\n\n", "\n");
	while (<$test_file>) {
		chomp;
		push @test_data, $_;
		shift @test_data;
		$centropy -= log2(ps($lambdas_ref, $test_data[2], $test_data[1], $test_data[0]));
		$test_trigrams_total++;
	}

	close $test_file
		or die "IO error while closing file '$test_file_name': $!\n";
	
	# Scale
	return $centropy / $test_trigrams_total;
}


# Calculate the cross entropy and print formatted output for plotting
print join("\t", @lambdas, cross_entropy(\@lambdas, $test_file_name)) . "\n";
