package org.apache.spark.ml.regression

import breeze.linalg.DenseVector
import breeze.linalg.functions.euclideanDistance
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasMaxIter, HasTol}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib.linalg.{Vectors => MLLibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

import org.apache.log4j.Logger
import org.apache.log4j.Level

trait LinearRegressionParams extends PredictorParams with HasMaxIter with HasTol {

    def numIter(value: Int): this.type = set(maxIter, value)

    final val lr: Param[Double] = new DoubleParam(
        this,
        "lr",
        "коэффициент обучения для алгоритма (LearningRate)"
    )

    def getLearningRate: Double = $(lr)
    setDefault(maxIter -> 1000, lr -> 0.05, tol -> 1e-7)

    protected def validateAndTransformSchema(Schema: StructType): StructType =
    {
        SchemaUtils.checkColumnType(Schema, getFeaturesCol, new VectorUDT())

        if (Schema.fieldNames.contains($(predictionCol))) {
            SchemaUtils.checkColumnType(Schema, getPredictionCol, new VectorUDT())
            Schema}

        else {SchemaUtils.appendColumn(Schema, Schema(getFeaturesCol).copy(name = getPredictionCol))
        }
    }
}

class LinearRegression_(
    override val uid: String
) extends Estimator[LRModel] with LinearRegressionParams with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("linearRegression_"))

    override def fit(dataset: Dataset[_]): LRModel = {

        implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
        val numEpochs = getMaxIter
        val learningRate = getLearningRate
        val tolerance = getTol
        val vectorAssembler = new VectorAssembler()
          .setInputCols(Array(getFeaturesCol, getLabelCol))
          .setOutputCol("result")
        val transformedData = vectorAssembler.transform(dataset)
          .select(col("result").as[Vector])
        val featureCount = transformedData.first().size - 1
        var previousWeights = DenseVector.fill(featureCount, Double.PositiveInfinity)
        val weights = DenseVector.fill(featureCount, 0.0)
        var iteration = 0
        while (iteration < numEpochs && euclideanDistance(weights.toDenseVector, previousWeights.toDenseVector) > tolerance) {
            iteration += 1
            val sum = transformedData.rdd.mapPartitions(data => {
                val summarizer = new MultivariateOnlineSummarizer()
                data.foreach(row => {
                    val features = row.asBreeze(0 until featureCount).toDenseVector
                    val label = row.asBreeze(-1)
                    val predictedLabel = features.dot(weights)
                    summarizer.add(MLLibVectors.fromBreeze((predictedLabel - label) * features))
                })
                Iterator(summarizer)
            }).reduce(_ merge _)

            previousWeights = weights.copy
            weights -= learningRate * sum.mean.asBreeze
            val epochLoss = euclideanDistance(weights.toDenseVector, previousWeights.toDenseVector)
            println(s"Epoch: $iteration, Loss: $epochLoss")
        }

        copyValues(new LRModel(Vectors.fromBreeze(weights)).setParent(this))
    }


    override def copy(extra: ParamMap): Estimator[LRModel] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

class LRModel(
    override val uid: String,
    val LRweights: Vector,
) extends Model[LRModel] with LinearRegressionParams {

    private[regression] def this(LRweights: Vector) = this(
        Identifiable.randomUID("LRModel_"),
        LRweights
    )

    override def transformSchema(schema: StructType): StructType = {
        var outputSchema = validateAndTransformSchema(schema)
        if ($(predictionCol).nonEmpty) {
            outputSchema = SchemaUtils.updateNumeric(outputSchema, $(predictionCol))
        }
        outputSchema
    }

    override def transform(dataset: Dataset[_]): DataFrame = {
        val predictUDF = udf { features: Any =>
            predict(features.asInstanceOf[Vector])
        }
        val outputSchema = transformSchema(dataset.schema)
        dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
    }


    override def copy(extra: ParamMap): LRModel = copyValues(
        new LRModel(LRweights), extra
    )
    private def predict(features: Vector) = features.asBreeze.dot(LRweights.asBreeze)

}
