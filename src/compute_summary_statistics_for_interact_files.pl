#! /usr/bin/perl -w

use strict;
use warnings;

use lib '/home/mering/lib/perl5/';
use JSON;

my %counts_per_organism_per_file = ();
my %counts_per_score_bin_per_organism_per_file = ();
my %nr_of_lines_seen_per_file = ();
my %nr_of_organisms_seen_per_file = ();
my %any_organism_seen = ();

my ($input_description_file) = @ARGV;

open (FH_R, "| R --vanilla");

die "usage: $0  input_description_file.json \n" unless ($input_description_file);

open (FH_INPUT, "$input_description_file") or die "cannot read file '$input_description_file'!\n";
my $json_input_string = "";
while (<FH_INPUT>) {
    $json_input_string .= $_;
}

my $input_description = from_json ($json_input_string);

die "error in parsing input_description_file '$input_description_file': JSON parsing problem !\n" unless defined $input_description;

my $run_title = $input_description->{run_title} or die "error: failed to parse required field 'run_title' ... check your json input file!\n";
my $output_filename_plots = $input_description->{output_filename_plots} or die "error: failed to parse required field 'output_filename_plots' ... check your json input file!\n";
my $output_filename_data = $input_description->{output_filename_data} or die "error: failed to parse required field 'output_filename_data' ... check your json input file!\n";
my $output_filename_errors = $input_description->{output_filename_errors} or die "error: failed to parse required field 'output_filename_errors' ... check your json input file!\n";
my $valid_proteins_file = $input_description->{valid_proteins_file} or die "error: failed to parse required field 'valid_proteins_file' ... check your json input file!\n";
my $benchmarking_file = $input_description->{benchmarking_file} or die "error: failed to parse required field 'benchmarking' ... check your json input file!\n"; 
my $redundancy_consider_first_interaction_only = $input_description->{redundancy_consider_first_interaction_only} or 0;
my $interactions_per_protein = $input_description->{interactions_per_protein} or 0;

my $nr_of_samples = 0;

my %files_to_compare = ();
my %color_of_file = ();
my %line_style_of_file = ();

foreach my $sample (@{$input_description->{samples}}) {
    my $name = $sample->{name};
    $color_of_file{$name} = $sample->{color};
    $line_style_of_file{$name} = $sample->{line};
    $files_to_compare{$name} = $sample->{data_file};
    $nr_of_samples += 1;
}

die "error: no samples in json input file !\n" unless $nr_of_samples > 0;

foreach my $file (sort keys %files_to_compare) {
    die "error: file '$files_to_compare{$file}' does not exist !\n" unless -e $files_to_compare{$file};
}

my %organisms_to_report = ();
my $organism_counter = 1;
foreach my $organism (@{$input_description->{organisms_to_report}}) {
    $organisms_to_report{$organism} = $organism_counter;
    $organism_counter += 1;
}

print STDERR "nr of samples to check: $nr_of_samples.\n";

$color_of_file{"random"} = "black";

my %valid_identifiers_per_taxon = ();
open (FH, $valid_proteins_file) or die "error: cannot read file '$valid_proteins_file'!\n";
while (<FH>) {
    chomp;
    my ($identifier, $taxon) = split /\t/;
    $valid_identifiers_per_taxon{$taxon}{$identifier} = 1;
}

open (FH_OUT, "> $output_filename_data") or die "error: cannot write to file '$output_filename_data'!\n";

print FH_OUT "#Input Files:\n";
foreach my $shorthand (sort keys %files_to_compare) {
    print FH_OUT "$shorthand\t$files_to_compare{$shorthand}\n";
}
print FH_OUT "\n";

print FH_OUT "#Organisms to report:\n";
foreach my $organism (sort {$organisms_to_report{$a} <=> $organisms_to_report{$b} } keys %organisms_to_report) {
    print FH_OUT "$organism\t$organisms_to_report{$organism}\n";
}
print FH_OUT "\n";

print FH_OUT "#Benchmarking file:\n";
print FH_OUT $benchmarking_file;
print FH_OUT "\n\n";

my %pathways_per_protein_in_benchmark = ();
my %eligible_benchmark_proteins_per_organism = ();

print STDERR "now reading benchmark file '$benchmarking_file' ... \n";

open (FH, $benchmarking_file) or die "cannot read file '$benchmarking_file'\n";
while (<FH>) {
    chomp;
    my ($organism, $pathway, $nr_of_proteins, $proteins_concatenated) = split /\t/;
    my @proteins = split /\s/, $proteins_concatenated;
    foreach my $protein (@proteins) {
        my $external_id = $organism . "." . $protein;
        $pathways_per_protein_in_benchmark{$external_id}{$pathway} = 1;
        $eligible_benchmark_proteins_per_organism{$organism}{$external_id} = 1;
    }
}

my %roc_data_per_file_per_organism = ();
my %min_score_per_organism_per_file = ();
my %max_score_per_organism_per_file = ();

my %parsing_errors_per_organism_per_type = ();

foreach my $file (sort keys %files_to_compare) {

    print STDERR "now reading file '$file' == '$files_to_compare{$file}' ... \n";
    my $sort = "";
    $sort = "sort -k5gr | " if $interactions_per_protein;

    if ($files_to_compare{$file} =~ /\.gz\z/) {
        open (FH, "gzip -cd $files_to_compare{$file} | $sort ") or die "cannot read input file '$files_to_compare{$file}'!\n";
    } else {
        open (FH, "cat $files_to_compare{$file} | $sort ") or die "cannot read input file '$files_to_compare{$file}'!\n";
    }

    my %duplet_counts = ();
    my %singleton_count = ();
    my %score_triplet_counts = ();
    my %evidence_triplet_counts = ();

    while (<FH>) {

        chomp; next if /\A\#/;

        my ($keyword, $organism, $identifier1, $identifier2, $score, $optional_evidence_1, $optional_evidence_2, $empty) = split /[\t ]/;

        next unless exists $valid_identifiers_per_taxon{$organism};

        $optional_evidence_1 = "empty" unless defined $optional_evidence_1;
        $optional_evidence_2 = "empty" unless defined $optional_evidence_2;

        if ($identifier1 eq $identifier2) {
            $parsing_errors_per_organism_per_type{$organism}{0} = "un-wanted self-interaction, skipped, in sample '$file': '$_'!" unless exists $parsing_errors_per_organism_per_type{$organism}{0};
            next;
        }

        unless (exists $valid_identifiers_per_taxon{$organism}{$identifier1}) {
            $parsing_errors_per_organism_per_type{$organism}{1} = "unknown identifier '$identifier1' in sample '$file': '$_'!" unless exists $parsing_errors_per_organism_per_type{$organism}{1};
        }
        unless (exists $valid_identifiers_per_taxon{$organism}{$identifier2}) {
            $parsing_errors_per_organism_per_type{$organism}{1} = "unknown identifier '$identifier2' in sample '$file': '$_'!" unless exists $parsing_errors_per_organism_per_type{$organism}{1};
        }

        my $protein1 = $organism . "." . $identifier1;
        my $protein2 = $organism . "." . $identifier2;

        my $score_triplet_forward = $protein1 . "+" . $protein2 . "+" . $score;
        my $score_triplet_reverse = $protein2 . "+" . $protein1 . "+" . $score;
        my $interaction_duplet = $protein1 . "+" . $protein2;
        my $evidence_triplet = $protein1 . "+" . $protein2 . "+" . $optional_evidence_1 . "+" . $optional_evidence_2;

        if ($redundancy_consider_first_interaction_only) {
            next if exists $duplet_counts{$interaction_duplet};
        }

        if ($interactions_per_protein) {
            $singleton_count{$protein1} = 0 unless exists $singleton_count{$protein1};
            next if $singleton_count{$protein1} >= $interactions_per_protein;
            $singleton_count{$protein1}++;
        } 

        $score_triplet_counts{$score_triplet_forward} += 1;
        $score_triplet_counts{$score_triplet_reverse} += 1;
        $duplet_counts{$interaction_duplet} += 1;
        $evidence_triplet_counts{$evidence_triplet} += 1;

        if (defined $empty) {
            $parsing_errors_per_organism_per_type{$organism}{2} = "parse warning, unexpected number of fields in sample '$file': line '$_'"
                unless exists $parsing_errors_per_organism_per_type{$organism}{2};
        }
        unless (defined $score) {
            $parsing_errors_per_organism_per_type{$organism}{3} = "parse warning, score not defined in sample '$file': line '$_'"
                unless exists $parsing_errors_per_organism_per_type{$organism}{3};
        }
        if ($score <= 0) {
            $parsing_errors_per_organism_per_type{$organism}{4} = "parse warning, score too low in sample '$file': line '$_'"
                unless exists $parsing_errors_per_organism_per_type{$organism}{4};
        }
        if ($score >= 1) {
            $parsing_errors_per_organism_per_type{$organism}{5} = "parse warning, score too high in sample '$file': line '$_'"
                unless exists $parsing_errors_per_organism_per_type{$organism}{5};
        }

        my $min_score = 1000000;
        my $max_score = -1000000;
        $min_score = $min_score_per_organism_per_file{$organism}{$file} if exists $min_score_per_organism_per_file{$organism}{$file};
        $max_score = $max_score_per_organism_per_file{$organism}{$file} if exists $max_score_per_organism_per_file{$organism}{$file};
        $min_score = $score if $score < $min_score;
        $max_score = $score if $score > $max_score;
        $min_score_per_organism_per_file{$organism}{$file} = $min_score;
        $max_score_per_organism_per_file{$organism}{$file} = $max_score;

        $score = int ($score * 1000);    ## this will round scores to no more than three digits after the comma (effectively binning them, for the score-count distribution plots).

        $counts_per_organism_per_file{$file}{$organism} += 1;
        $counts_per_score_bin_per_organism_per_file{$file}{$organism}{$score} += 1;
        $counts_per_score_bin_per_organism_per_file{$file}{'total'}{$score} += 1;
        $nr_of_lines_seen_per_file{$file} += 1;
        $any_organism_seen{$organism} = 1;

        if (exists $organisms_to_report{$organism}) {
            if (exists $pathways_per_protein_in_benchmark{$protein1}) {
                if (exists $pathways_per_protein_in_benchmark{$protein2}) {
                    my $status = "false";
                    foreach my $pathway1 (keys %{$pathways_per_protein_in_benchmark{$protein1}}) {
                        $status = "true" if exists $pathways_per_protein_in_benchmark{$protein2}{$pathway1};
                    }
                    my $random_score = $score + rand () / 10;   ## to make sure a random sort order is enforced for the ROC curve
                    $roc_data_per_file_per_organism{$file}{$organism}{$random_score} = $status;
                }
            }
        }
    }

    unless ($interactions_per_protein) {
        foreach my $score_triple (keys %score_triplet_counts) {
            my $count = $score_triplet_counts{$score_triple};
            if ($count % 2) {
                my ($organism) = $score_triple =~ /\A(\d+)\./;
                $parsing_errors_per_organism_per_type{$organism}{6} = "reciprocality seems violated in sample '$file', for triple '$score_triple'"
                    unless exists $parsing_errors_per_organism_per_type{$organism}{6};
            }
        }
    } else {
        foreach my $organism (keys %any_organism_seen) {
            $parsing_errors_per_organism_per_type{$organism}{6} = "reciprocality can't be asserted for '$file' while interactions_per_protein parameter is provided";
        }
    }

    foreach my $evidence_triple (keys %evidence_triplet_counts) {
        my $count = $evidence_triplet_counts{$evidence_triple};
        if ($count > 1) {
            my ($organism) = $evidence_triple =~ /\A(\d+)\./;
            $parsing_errors_per_organism_per_type{$organism}{7} = "same evidence for same edge in '$file', for triple '$evidence_triple'"
                unless exists $parsing_errors_per_organism_per_type{$organism}{7};
        }
    }

    foreach my $interaction_duplet (keys %duplet_counts) {
        my $count = $duplet_counts{$interaction_duplet};
        if ($count > 1) {
            my ($organism) = $interaction_duplet =~ /\A(\d+)\./;
            $parsing_errors_per_organism_per_type{$organism}{8} = "same edge occurs multiple times in '$file', for interaction '$interaction_duplet'"
                unless exists $parsing_errors_per_organism_per_type{$organism}{8};
        }
    }

    $nr_of_organisms_seen_per_file{$file} = scalar keys %{$counts_per_organism_per_file{$file}};

    close FH;
}

open (FH_ERRORS, "> $output_filename_errors") or die "cannot write to file: '$output_filename_errors'!\n";
foreach my $organism (sort {$a <=> $b} keys %parsing_errors_per_organism_per_type) {
    next unless defined $parsing_errors_per_organism_per_type{$organism};
    print FH_ERRORS "= = = errors in input file = = = organism '$organism'\n";
    foreach my $type (keys %{$parsing_errors_per_organism_per_type{$organism}}) {
        print FH_ERRORS $parsing_errors_per_organism_per_type{$organism}{$type};
        print FH_ERRORS "\n";
    }
}
close FH_ERRORS;

my $column_header_simple = join "\t", sort keys %files_to_compare;
my $column_header_double = join "\t", sort (keys %files_to_compare, keys %files_to_compare);

print FH_OUT "\n\n## total number of lines:\n";
print FH_OUT "$column_header_simple\n";
foreach my $file (sort keys %files_to_compare) { print FH_OUT "$nr_of_lines_seen_per_file{$file}\t"; };

print FH_OUT "\n\n## nr of organisms contained:\n";
print FH_OUT "$column_header_simple\n";
foreach my $file (sort keys %files_to_compare) { print FH_OUT "$nr_of_organisms_seen_per_file{$file}\t"; };

print FH_OUT "\n\n## organism counts:\n";
print FH_OUT "organism\t$column_header_double\n";

foreach my $organism (sort {$a <=> $b} keys %any_organism_seen) {
    print FH_OUT "$organism";
    foreach my $file (sort keys %files_to_compare) {
        my $raw_count = 0;
        $raw_count = $counts_per_organism_per_file{$file}{$organism} if exists $counts_per_organism_per_file{$file}{$organism};
        my $normalized_count = $raw_count / $nr_of_lines_seen_per_file{$file};
        print FH_OUT "\t$raw_count\t$normalized_count";
    }
    print FH_OUT "\n";
}

my %score_counts_per_organism_per_file_per_bin = ();
my %max_score_count_per_organism = ();
my %max_score_bin_per_organism = ();
my %min_score_bin_per_organism = ();

foreach my $organism (sort {$organisms_to_report{$a} <=> $organisms_to_report{$b}} keys %organisms_to_report) {

    next unless exists $any_organism_seen{$organism};

    $max_score_bin_per_organism{$organism} = -1000;
    $min_score_bin_per_organism{$organism} = 1000;
    $max_score_count_per_organism{$organism} = 0;

    print FH_OUT "\n\n## score bin counts ($organism):\n";

    print FH_OUT "score\t$column_header_double\n";

    my %available_score_bins = ();
    foreach my $file (sort keys %files_to_compare) {
        foreach my $score_bin (keys %{$counts_per_score_bin_per_organism_per_file{$file}{$organism}}) {
            $available_score_bins{$score_bin} = 1;
        }
    }

    foreach my $score_bin (sort {$a <=> $b} keys %available_score_bins) {

        my $actual_score = $score_bin / 1000;
        print FH_OUT $actual_score;

        foreach my $file (sort keys %files_to_compare) {

            my $raw_count = 0;
            $raw_count = $counts_per_score_bin_per_organism_per_file{$file}{$organism}{$score_bin} if exists $counts_per_score_bin_per_organism_per_file{$file}{$organism}{$score_bin};
            my $normalized_count = $raw_count / $nr_of_lines_seen_per_file{$file};
            print FH_OUT "\t$raw_count\t$normalized_count";
            $score_counts_per_organism_per_file_per_bin{$organism}{$file}{$actual_score} = $raw_count;
            $max_score_bin_per_organism{$organism} = $actual_score if $actual_score > $max_score_bin_per_organism{$organism};
            $min_score_bin_per_organism{$organism} = $actual_score if $actual_score < $min_score_bin_per_organism{$organism};
            $max_score_count_per_organism{$organism} = $raw_count if $raw_count > $max_score_count_per_organism{$organism};
        }

        print FH_OUT "\n";
    }
}

my %roc_data_per_organism_per_file_x = ();
my %roc_data_per_organism_per_file_y = ();
my %roc_data_max_false_positive_count_per_organism = ();
my %roc_data_max_true_positive_count_per_organism = ();

foreach my $organism (sort {$organisms_to_report{$a} <=> $organisms_to_report{$b}} keys %organisms_to_report) {

    next unless exists $any_organism_seen{$organism};

    next if $organism eq "total";

    print FH_OUT "\n\n## roc-curve ($organism):\n";
    print FH_OUT "false_positives\t$column_header_simple\n";
    my $tab_counter = 0;

    foreach my $file (sort keys %files_to_compare) {

        $tab_counter += 1;
        my $false_positive_counter = 0;
        my $true_positive_counter = 0;
        my $skip_multiplier = 1;
        my $skip_counter = 1;
        foreach my $interaction (sort {$b <=> $a} keys %{$roc_data_per_file_per_organism{$file}{$organism}}) {
            if ($roc_data_per_file_per_organism{$file}{$organism}{$interaction} eq "true") {
                $true_positive_counter += 1;
            } else {
                $false_positive_counter += 1;
            }
            if ($skip_counter > 0) {
                $skip_counter -= 1;
                next;
            }
            $skip_multiplier *= 1.5;
            $skip_counter = int ($skip_multiplier);
            print FH_OUT $false_positive_counter;
            foreach my $count (1..$tab_counter) {
                print FH_OUT "\t";
            }
            print FH_OUT $true_positive_counter;
            print FH_OUT "\n";
            push @{$roc_data_per_organism_per_file_x{$organism}{$file}}, $false_positive_counter;
            push @{$roc_data_per_organism_per_file_y{$organism}{$file}}, $true_positive_counter;
        }

        ## print the very last datapoint:

        print FH_OUT $false_positive_counter;
        foreach my $count (1..$tab_counter) {
            print FH_OUT "\t";
        }
        print FH_OUT $true_positive_counter;
        print FH_OUT "\n";
        push @{$roc_data_per_organism_per_file_x{$organism}{$file}}, $false_positive_counter;
        push @{$roc_data_per_organism_per_file_y{$organism}{$file}}, $true_positive_counter;

        my $max_false_positives_this_organism = 0;
        $max_false_positives_this_organism = $roc_data_max_false_positive_count_per_organism{$organism} if exists $roc_data_max_false_positive_count_per_organism{$organism};
        $max_false_positives_this_organism = $false_positive_counter if $false_positive_counter > $max_false_positives_this_organism;
        $roc_data_max_false_positive_count_per_organism{$organism} = $max_false_positives_this_organism;

        my $max_true_positives_this_organism = 0;
        $max_true_positives_this_organism = $roc_data_max_true_positive_count_per_organism{$organism} if exists $roc_data_max_true_positive_count_per_organism{$organism};
        $max_true_positives_this_organism = $true_positive_counter if $true_positive_counter > $max_true_positives_this_organism;
        $roc_data_max_true_positive_count_per_organism{$organism} = $max_true_positives_this_organism;
    }
}

## create random roc performance, per organism

my %random_roc_data_per_organism_per_file_x = ();
my %random_roc_data_per_organism_per_file_y = ();

foreach my $organism (sort {$organisms_to_report{$a} <=> $organisms_to_report{$b}} keys %organisms_to_report) {

    next unless exists $any_organism_seen{$organism};

    next if $organism eq "total";

    my @eligible_proteins = keys %{$eligible_benchmark_proteins_per_organism{$organism}};
    my $nr_eligible_proteins = scalar @eligible_proteins;

    my $max_false_positives_to_report = 0;
    $max_false_positives_to_report = $roc_data_max_false_positive_count_per_organism{$organism} if exists $roc_data_max_false_positive_count_per_organism{$organism};

    my $false_positive_counter = 0;
    my $true_positive_counter = 0;
    my $skip_multiplier = 1;
    my $skip_counter = 1;

    while ($false_positive_counter < $max_false_positives_to_report) {

        my $index_a = int (rand ($nr_eligible_proteins));
        my $index_b = int (rand ($nr_eligible_proteins));

        next if $index_a == $index_b;

        my $protein_a = $eligible_proteins[$index_a];
        my $protein_b = $eligible_proteins[$index_b];

        next unless exists $pathways_per_protein_in_benchmark{$protein_a};
        next unless exists $pathways_per_protein_in_benchmark{$protein_b};

        my $status = "false";
        foreach my $pathway1 (keys %{$pathways_per_protein_in_benchmark{$protein_a}}) {
            $status = "true" if exists $pathways_per_protein_in_benchmark{$protein_b}{$pathway1};
        }

        if ($status eq "true") {
            $true_positive_counter += 1;
        } else {
            $false_positive_counter += 1;
        }
        if ($skip_counter > 0) {
            $skip_counter -= 1;
            next;
        }
        $skip_multiplier *= 1.5;
        $skip_counter = int ($skip_multiplier);
        push @{$random_roc_data_per_organism_per_file_x{$organism}}, $false_positive_counter;
        push @{$random_roc_data_per_organism_per_file_y{$organism}}, $true_positive_counter;
    }

    ## record the very last datapoint:

    push @{$random_roc_data_per_organism_per_file_x{$organism}}, $false_positive_counter;
    push @{$random_roc_data_per_organism_per_file_y{$organism}}, $true_positive_counter;
}

## compute score equivalence

my %score_equivalence_per_organism_per_file_per_bin = ();

foreach my $organism (sort {$organisms_to_report{$a} <=> $organisms_to_report{$b}} keys %organisms_to_report) {

    next unless exists $any_organism_seen{$organism};

    next if $organism eq "total";

    my $nr_of_bins = 20;

    print FH_OUT "\n\n## score-comparison ($organism):\n";
    print FH_OUT "input_score\t$column_header_simple\n";

    my $tab_counter = 0;

    foreach my $file (sort keys %files_to_compare) {

        $tab_counter += 1;

        my $nr_of_interactions = scalar keys %{$roc_data_per_file_per_organism{$file}{$organism}};

        my %score_sum_per_bin = ();
        my %true_positives_per_bin = ();
        my %false_positives_per_bin = ();

        my $max_score = $max_score_per_organism_per_file{$organism}{$file};
        my $min_score = $min_score_per_organism_per_file{$organism}{$file};

        foreach my $interaction (sort {$b <=> $a} keys %{$roc_data_per_file_per_organism{$file}{$organism}}) {

            my $score = $interaction / 1000.0;   # (!)

            my $bin = 1;
            if ($max_score > $min_score) {
                $bin = int ((($score - $min_score) / ($max_score - $min_score)) * $nr_of_bins);
            }

            $score_sum_per_bin{$bin} += $score;

            if ($roc_data_per_file_per_organism{$file}{$organism}{$interaction} eq "true") {
                $true_positives_per_bin{$bin} += 1;
            } else {
                $false_positives_per_bin{$bin} += 1;
            }
        }

        foreach my $bin (sort {$a <=> $b} keys %score_sum_per_bin) {

            my $true_positives = 0; $true_positives = $true_positives_per_bin{$bin} if exists $true_positives_per_bin{$bin};
            my $false_positives = 0; $false_positives = $false_positives_per_bin{$bin} if exists $false_positives_per_bin{$bin};

            my $nr_interactions_this_bin = $true_positives + $false_positives;

            next unless $nr_interactions_this_bin >= 20;

            my $score_sum = $score_sum_per_bin{$bin};

            my $input_score = $score_sum / $nr_interactions_this_bin;

            my $benchmarked_score = $true_positives / $nr_interactions_this_bin;

            print FH_OUT "$bin\t$input_score";
            foreach my $count (1..$tab_counter) {
                print FH_OUT "\t";
            }
            print FH_OUT $benchmarked_score;
            print FH_OUT "\n";
            $score_equivalence_per_organism_per_file_per_bin{$organism}{$file}{$input_score} = $benchmarked_score;
        }
    }
}

## create R plots in pdf.

my $R_command = "pdf (\"$output_filename_plots\")\n".
    "par (mfrow=c(2,2), cex=0.5)\n".
    "options (scipen=10)\n";

foreach my $organism (sort {$organisms_to_report{$a} <=> $organisms_to_report{$b}} keys %organisms_to_report) {

    next unless exists $any_organism_seen{$organism};

    next unless exists $roc_data_max_true_positive_count_per_organism{$organism};

    ## first, a normal ROC plot.

    $R_command .= "plot (x=NULL, y=NULL, ".
        "xlim=c(0,$roc_data_max_false_positive_count_per_organism{$organism}), ".
        "ylim=c(0,$roc_data_max_true_positive_count_per_organism{$organism}), xlab=\"false positives\", ylab=\"true positives\", main=\"$run_title (organism $organism)\")\n";

    my $y_offset_legend = 0;

    foreach my $file ("random", sort keys %files_to_compare) {

        my $x_vector = $roc_data_per_organism_per_file_x{$organism}{$file};
        my $y_vector = $roc_data_per_organism_per_file_y{$organism}{$file};
        my $line_type = ($line_style_of_file{$file} or "solid");
        if ($file eq "random") {
            $x_vector = $random_roc_data_per_organism_per_file_x{$organism};
            $y_vector = $random_roc_data_per_organism_per_file_y{$organism};
            $line_type = "dotted";
        }

        $R_command .= "false_vector = c(" . join ",", @$x_vector;
        $R_command .= ")\n";
        $R_command .= "true_vector = c(" . join ",", @$y_vector;
        $R_command .= ")\n";

        my $y_offset = $y_offset_legend * $roc_data_max_true_positive_count_per_organism{$organism};
        my $x_offset = 0.8 * $roc_data_max_false_positive_count_per_organism{$organism};

        $R_command .= "points (false_vector, true_vector, col=\"$color_of_file{$file}\")\n";
        $R_command .= "lines (false_vector, true_vector, lty=\"$line_type\", col=\"$color_of_file{$file}\")\n";
        $R_command .= "text ($x_offset, $y_offset, \"$file\", col=\"$color_of_file{$file}\")\n";

        $y_offset_legend += 0.03;
    }

    ## now, same ROC plot with log/log axes

    $R_command .= "plot (x=c(1,1), y=c(1,1), log=\"xy\", ".
        "xlim=c(1,$roc_data_max_false_positive_count_per_organism{$organism}), ".
        "ylim=c(1,$roc_data_max_true_positive_count_per_organism{$organism}), xlab=\"false positives\", ylab=\"true positives\", main=\"$run_title (organism $organism, log/log)\")\n";

    $y_offset_legend = 1;

    foreach my $file ("random", sort keys %files_to_compare) {

        my $x_vector = $roc_data_per_organism_per_file_x{$organism}{$file};
        my $y_vector = $roc_data_per_organism_per_file_y{$organism}{$file};
        my $line_type = ($line_style_of_file{$file} or "solid");
        if ($file eq "random") {
            $x_vector = $random_roc_data_per_organism_per_file_x{$organism};
            $y_vector = $random_roc_data_per_organism_per_file_y{$organism};
            $line_type = "dotted";
        }

        $R_command .= "false_vector = c(" . join ",", @$x_vector;
        $R_command .= ")\n";
        $R_command .= "true_vector = c(" . join ",", @$y_vector;
        $R_command .= ")\n";

        my $y_offset = $y_offset_legend;
        my $x_offset = 0.1 * $roc_data_max_false_positive_count_per_organism{$organism};

        $R_command .= "points (false_vector, true_vector, col=\"$color_of_file{$file}\")\n";
        $R_command .= "lines (false_vector, true_vector, lty=\"$line_type\", col=\"$color_of_file{$file}\")\n";
        $R_command .= "text ($x_offset, $y_offset, \"$file\", col=\"$color_of_file{$file}\")\n";

        $y_offset_legend *= 1 + (log ($roc_data_max_true_positive_count_per_organism{$organism}) / 30);
    }

    ## frequency plot showing the score distribution

    $R_command .= "plot (x=c(0,0), y=c(1,1), log=\"y\", col=\"white\", ".
        "xlim=c($min_score_bin_per_organism{$organism},$max_score_bin_per_organism{$organism}), ".
        "ylim=c(1,$max_score_count_per_organism{$organism}), xlab=\"score\", ylab=\"counts\", main=\"score distribution (organism $organism)\")\n";

    $y_offset_legend = 1;

    foreach my $file (sort keys %files_to_compare) {

        my @x_vector = ();
        my @y_vector = ();

        foreach my $score_bin (sort {$a <=> $b} keys %{$score_counts_per_organism_per_file_per_bin{$organism}{$file}}) {
            push @x_vector, $score_bin;
            push @y_vector, $score_counts_per_organism_per_file_per_bin{$organism}{$file}{$score_bin};
        }

        $R_command .= "false_vector = c(" . join ",", @x_vector;
        $R_command .= ")\n";
        $R_command .= "true_vector = c(" . join ",", @y_vector;
        $R_command .= ")\n";

        my $y_offset = $y_offset_legend;
        my $x_offset = 0.8 * $max_score_bin_per_organism{$organism};

        $R_command .= "points (false_vector, true_vector, col=\"$color_of_file{$file}\")\n";
        $R_command .= "text ($x_offset, $y_offset, \"$file\", col=\"$color_of_file{$file}\")\n";

        $y_offset_legend *= 1 + (log ($roc_data_max_true_positive_count_per_organism{$organism}) / 30);
    }

    ## score equivalence plot

    $R_command .= "plot (x=NULL, y=NULL, ".
        "xlim=c($min_score_bin_per_organism{$organism},$max_score_bin_per_organism{$organism}), ".
        "ylim=c(0,1), xlab=\"input score (binned)\", ylab=\"actual benchmarked performance\", main=\"score correctness (organism $organism)\")\n";

    $y_offset_legend = 0;

    foreach my $file (sort keys %files_to_compare) {

        my @x_vector = ();
        my @y_vector = ();

        foreach my $score_bin (sort {$a <=> $b} keys %{$score_equivalence_per_organism_per_file_per_bin{$organism}{$file}}) {
            push @x_vector, $score_bin;
            push @y_vector, $score_equivalence_per_organism_per_file_per_bin{$organism}{$file}{$score_bin};
        }

        my $line_type = ($line_style_of_file{$file} or "solid");

        $R_command .= "false_vector = c(" . join ",", @x_vector;
        $R_command .= ")\n";
        $R_command .= "true_vector = c(" . join ",", @y_vector;
        $R_command .= ")\n";

        my $y_offset = $y_offset_legend;
        my $x_offset = 0.8 * $max_score_bin_per_organism{$organism};

        $R_command .= "points (false_vector, true_vector, col=\"$color_of_file{$file}\")\n";
        $R_command .= "lines (false_vector, true_vector, lty=\"$line_type\", col=\"$color_of_file{$file}\")\n";
        $R_command .= "text ($x_offset, $y_offset, \"$file\", col=\"$color_of_file{$file}\")\n";

        $y_offset_legend += 0.03;
    }
}

close FH_OUT;

print STDERR "generating pdfs (using R) ... \n";

print FH_R $R_command;
close FH_R;

print STDERR "all done.\n";
