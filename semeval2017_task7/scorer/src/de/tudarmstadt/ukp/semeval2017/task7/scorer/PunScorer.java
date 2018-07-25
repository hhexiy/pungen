/*
 * Copyright 2016
 * Ubiquitous Knowledge Processing (UKP) Lab
 * Technische Universität Darmstadt
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.tudarmstadt.ukp.semeval2017.task7.scorer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.text.ParseException;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * @author Tristan Miller <miller@ukp.informatik.tu-darmstadt.de>
 *
 */
public class PunScorer {

	static class Pun {
		Set<String> sense1;
		Set<String> sense2;

		public Pun(String sense1, String sense2) {
			this.sense1 = new TreeSet<String>(Arrays.asList(sense1.split(";")));
			this.sense2 = new TreeSet<String>(Arrays.asList(sense2.split(";")));
		}

		/**
		 * Determines equality of two pun sense assignments based on minimal
		 * match agreement. That is, each of the sense sets assigned by the
		 * annotator is a subset of one (and only one) of the sense sets from
		 * the gold standard.
		 */
		@Override
		public boolean equals(Object resultPun) {
			if (!(resultPun instanceof Pun)) {
				return false;
			}

			Set<String> goldSense1 = this.sense1;
			Set<String> goldSense2 = this.sense2;
			Set<String> resultSense1 = ((Pun) resultPun).sense1;
			Set<String> resultSense2 = ((Pun) resultPun).sense2;

			if (resultSense1.isEmpty() || resultSense2.isEmpty())
				return false;

			if (goldSense1.containsAll(resultSense1) && goldSense2.containsAll(resultSense2))
				return true;

			if (goldSense1.containsAll(resultSense2) && goldSense2.containsAll(resultSense1))
				return true;

			return false;
		}

	}

	/**
	 * @param args
	 * @throws ParseException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException, ParseException {
		if (args.length < 3 || args.length > 4) {
			printHelpAndExit();
		}

		String subtask = args[0];
		String goldFile = args[1];
		String resultFile = args[2];
		BufferedWriter outputWriter;

		if (args.length == 3) {
			outputWriter = new BufferedWriter(new OutputStreamWriter(System.out));
		} else {
			String outputFile = args[3];
			outputWriter = new BufferedWriter(new FileWriter(outputFile));
		}

		if (subtask.equals("-d")) {
			scoreClassificationTask(readBooleanMap(goldFile), readBooleanMap(resultFile), outputWriter);
		} else if (subtask.equals("-l")) {
			scoreDisambiguationTask(readStringMap(goldFile), readStringMap(resultFile), outputWriter);
		} else if (subtask.equals("-i")) {
			scoreDisambiguationTask(readPunMap(goldFile), readPunMap(resultFile), outputWriter);
		} else {
			printHelpAndExit();
		}

		outputWriter.close();

	}

	/**
	 * Scores task using WSD-style metrics
	 *
	 * @param goldFile
	 * @param resultFile
	 * @param outputWriter
	 * @throws IOException
	 * @throws ParseException
	 */
	private static <K> void scoreDisambiguationTask(Map<String, K> goldKey, Map<String, K> resultKey,
			BufferedWriter outputWriter) throws IOException, ParseException {
		int guessed = 0, correct = 0;

		for (String id : resultKey.keySet()) {
			if (!goldKey.containsKey(id)) {
				throw new IllegalArgumentException("Result ID '" + id + "' not in gold key");
			}
		}

		for (String id : goldKey.keySet()) {
			if (resultKey.containsKey(id)) {
				guessed++;
				if (goldKey.get(id).equals(resultKey.get(id))) {
					correct++;
				}
			}
		}

		outputDisambiguationScores(guessed, correct, goldKey.size(), outputWriter);
	}

	/**
	 * Outputs disambiguation scores
	 *
	 * @param guessed
	 * @param correct
	 * @param total
	 * @param outputWriter
	 * @throws IOException
	 */
	private static void outputDisambiguationScores(int guessed, int correct, int total, BufferedWriter outputWriter)
			throws IOException {
		double coverage = 0.0, precision = 0.0, recall = 0.0, f1 = 0.0;

		if (total > 0) {
			coverage = (double) guessed / total;
			recall = (double) correct / total;
		}

		if (guessed > 0) {
			precision = (double) correct / guessed;
		}

		if (precision + recall > 0.0) {
			f1 = (2 * precision * recall) / (precision + recall);
		}

		outputWriter.write("coverage: " + coverage);
		outputWriter.newLine();

		outputWriter.write("precision: " + precision);
		outputWriter.newLine();

		outputWriter.write("recall: " + recall);
		outputWriter.newLine();

		outputWriter.write("f1: " + f1);
		outputWriter.newLine();
	}

	/**
	 * Scores task using standard classification metrics
	 *
	 * @param goldFile
	 * @param resultFile
	 * @param outputWriter
	 * @throws IOException
	 * @throws ParseException
	 */
	private static void scoreClassificationTask(Map<String, Boolean> goldKey, Map<String, Boolean> resultKey,
			BufferedWriter outputWriter) throws IOException, ParseException {
		int tp = 0, tn = 0, fp = 0, fn = 0;

		for (String id : resultKey.keySet()) {
			if (!goldKey.containsKey(id)) {
				throw new IllegalArgumentException("Result ID '" + id + "' not in gold key");
			}
		}

		for (String id : goldKey.keySet()) {
			if (!resultKey.containsKey(id)) {
				System.err.println("Gold ID '" + id + "' missing from result key");
			}
			if (goldKey.get(id).equals(true) && resultKey.get(id).equals(true)) {
				tp++;
			} else if (goldKey.get(id).equals(true) && resultKey.get(id).equals(false)) {
				fn++;
			} else if (goldKey.get(id).equals(false) && resultKey.get(id).equals(true)) {
				fp++;
			} else if (goldKey.get(id).equals(false) && resultKey.get(id).equals(false)) {
				tn++;
			}
		}

		outputClassificationScores(tp, tn, fp, fn, outputWriter);
	}

	/**
	 * Outputs classification scores
	 * 
	 * @param tp
	 * @param tn
	 * @param fp
	 * @param fn
	 * @param outputWriter
	 * @throws IOException
	 */
	private static void outputClassificationScores(int tp, int tn, int fp, int fn, BufferedWriter outputWriter)
			throws IOException {
		double precision = 0.0, recall = 0.0, accuracy = 0.0, f1 = 0.0;

		if (tp + fp > 0) {
			precision = (double) tp / (tp + fp);
		}

		if (tp + fn > 0) {
			recall = (double) tp / (tp + fn);
		}

		if (tp + fp + tn + fn > 0) {
			accuracy = (double) (tp + tn) / (tp + fp + tn + fn);
		}

		if (precision + recall > 0.0) {
			f1 = (2 * precision * recall) / (precision + recall);
		}

		outputWriter.write("precision: " + precision);
		outputWriter.newLine();

		outputWriter.write("recall: " + recall);
		outputWriter.newLine();

		outputWriter.write("accuracy: " + accuracy);
		outputWriter.newLine();

		outputWriter.write("f1: " + f1);
		outputWriter.newLine();
	}

	/**
	 * Print usage instructions and exit
	 */
	private static void printHelpAndExit() {
		StackTraceElement[] stack = Thread.currentThread().getStackTrace();
		StackTraceElement main = stack[stack.length - 1];
		String mainClass = main.getClassName();
		System.err.println(
				"Usage:\n" + "\tjava " + mainClass + " [ -d | -l | -i ] <goldFile> <resultFile> [ <outputFile> ]");
		System.exit(1);
	}

	/**
	 * Reads a two-column, whitespace-delimited text file into a
	 * String-to-Boolean map
	 *
	 * @param filename
	 * @return
	 * @throws ParseException
	 * @throws IOException
	 */
	private static Map<String, Boolean> readBooleanMap(String filename) throws IOException, ParseException {
		Map<String, Boolean> booleanMap = new TreeMap<String, Boolean>();

		BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
		String line;
		int lineNumber = 0;

		while ((line = bufferedReader.readLine()) != null) {
			lineNumber++;
			String[] lineParts = line.split("[\\t ]");
			if (lineParts.length != 2) {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Syntax error on line " + lineNumber + " of file " + filename + ": invalid field count",
						lineNumber);
			}
			if (booleanMap.containsKey(lineParts[0])) {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Duplicate ID '" + lineParts[0] + "' on line " + lineNumber + " of file " + filename,
						lineNumber);
			}
			if (lineParts[1].equals("0")) {
				booleanMap.put(lineParts[0], false);
			} else if (lineParts[1].equals("1")) {
				booleanMap.put(lineParts[0], true);
			} else {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Syntax error on line " + lineNumber + " of file " + filename + ": value must be 0 or 1",
						lineNumber);
			}
		}
		bufferedReader.close();

		return booleanMap;
	}

	/**
	 * Reads a two-column, whitespace-delimited text file into a
	 * String-to-String map
	 *
	 * @param filename
	 * @return
	 * @throws ParseException
	 * @throws IOException
	 */
	private static Map<String, String> readStringMap(String filename) throws IOException, ParseException {
		Map<String, String> stringMap = new TreeMap<String, String>();

		BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
		String line;
		int lineNumber = 0;

		while ((line = bufferedReader.readLine()) != null) {
			lineNumber++;
			String[] lineParts = line.split("[\\t ]+");
			if (lineParts.length != 2) {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Syntax error on line " + lineNumber + " of file " + filename + ": invalid field count",
						lineNumber);
			}
			if (stringMap.containsKey(lineParts[0])) {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Duplicate ID '" + lineParts[0] + "' on line " + lineNumber + " of file " + filename,
						lineNumber);
			} else {
				stringMap.put(lineParts[0], lineParts[1]);
			}

		}
		bufferedReader.close();

		return stringMap;
	}

	/**
	 * Reads a three-column, whitespace-delimited text file into a String-to-Pun
	 * map
	 *
	 * @param filename
	 * @return
	 * @throws ParseException
	 * @throws IOException
	 */
	private static Map<String, Pun> readPunMap(String filename) throws IOException, ParseException {
		Map<String, Pun> punMap = new TreeMap<String, Pun>();

		BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
		String line;
		int lineNumber = 0;

		while ((line = bufferedReader.readLine()) != null) {
			lineNumber++;
			String[] lineParts = line.split("[\\t ]");
			if (lineParts.length != 3) {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Syntax error on line " + lineNumber + " of file " + filename + ": invalid field count",
						lineNumber);
			}
			if (punMap.containsKey(lineParts[0])) {
				bufferedReader.close();
				throw new java.text.ParseException(
						"Duplicate ID '" + lineParts[0] + "' on line " + lineNumber + " of file " + filename,
						lineNumber);
			} else {
				punMap.put(lineParts[0], new Pun(lineParts[1], lineParts[2]));
			}

		}
		bufferedReader.close();

		return punMap;
	}

}
