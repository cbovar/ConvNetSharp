namespace SlotMachineDemo
{
    partial class TrainingForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(TrainingForm));
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.btnRunResume = new System.Windows.Forms.Button();
            this.btnLearning = new System.Windows.Forms.Button();
            this.textBoxInformation = new System.Windows.Forms.TextBox();
            this.textBoxActions = new System.Windows.Forms.TextBox();
            this.chartAvReward = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.btnSave = new System.Windows.Forms.Button();
            this.btnLoad = new System.Windows.Forms.Button();
            this.chartAvLoss = new System.Windows.Forms.DataVisualization.Charting.Chart();
            ((System.ComponentModel.ISupportInitialize)(this.chartAvReward)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartAvLoss)).BeginInit();
            this.SuspendLayout();
            // 
            // btnRunResume
            // 
            resources.ApplyResources(this.btnRunResume, "btnRunResume");
            this.btnRunResume.Name = "btnRunResume";
            this.btnRunResume.UseVisualStyleBackColor = true;
            this.btnRunResume.Click += new System.EventHandler(this.OnRunResume);
            // 
            // btnLearning
            // 
            resources.ApplyResources(this.btnLearning, "btnLearning");
            this.btnLearning.Name = "btnLearning";
            this.btnLearning.UseVisualStyleBackColor = true;
            this.btnLearning.Click += new System.EventHandler(this.OnStartStopLearning);
            // 
            // textBoxInformation
            // 
            resources.ApplyResources(this.textBoxInformation, "textBoxInformation");
            this.textBoxInformation.Name = "textBoxInformation";
            this.textBoxInformation.ReadOnly = true;
            // 
            // textBoxActions
            // 
            resources.ApplyResources(this.textBoxActions, "textBoxActions");
            this.textBoxActions.Name = "textBoxActions";
            this.textBoxActions.ReadOnly = true;
            // 
            // chartAvReward
            // 
            this.chartAvReward.AccessibleRole = System.Windows.Forms.AccessibleRole.None;
            chartArea1.AxisX.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea1.AxisX2.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea1.AxisY.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea1.AxisY2.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea1.BorderColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea1.Name = "chartArea";
            this.chartAvReward.ChartAreas.Add(chartArea1);
            legend1.Alignment = System.Drawing.StringAlignment.Center;
            legend1.DockedToChartArea = "chartArea";
            legend1.Docking = System.Windows.Forms.DataVisualization.Charting.Docking.Left;
            legend1.Enabled = false;
            legend1.LegendStyle = System.Windows.Forms.DataVisualization.Charting.LegendStyle.Row;
            legend1.Name = "legend";
            this.chartAvReward.Legends.Add(legend1);
            resources.ApplyResources(this.chartAvReward, "chartAvReward");
            this.chartAvReward.Name = "chartAvReward";
            series1.BorderWidth = 2;
            series1.ChartArea = "chartArea";
            series1.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series1.Color = System.Drawing.Color.FromArgb(((int)(((byte)(204)))), ((int)(((byte)(37)))), ((int)(((byte)(41)))));
            series1.Legend = "legend";
            series1.LegendText = "Average Reward";
            series1.Name = "seriesReward";
            this.chartAvReward.Series.Add(series1);
            this.chartAvReward.Series.SuspendUpdates();
            // 
            // btnSave
            // 
            resources.ApplyResources(this.btnSave, "btnSave");
            this.btnSave.Name = "btnSave";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.OnSave);
            // 
            // btnLoad
            // 
            resources.ApplyResources(this.btnLoad, "btnLoad");
            this.btnLoad.Name = "btnLoad";
            this.btnLoad.UseVisualStyleBackColor = true;
            this.btnLoad.Click += new System.EventHandler(this.OnLoad);
            // 
            // chartAvLoss
            // 
            this.chartAvLoss.AccessibleRole = System.Windows.Forms.AccessibleRole.None;
            chartArea2.AxisX.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea2.AxisX2.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea2.AxisY.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea2.AxisY2.LineColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea2.BorderColor = System.Drawing.Color.FromArgb(((int)(((byte)(83)))), ((int)(((byte)(81)))), ((int)(((byte)(84)))));
            chartArea2.Name = "chartArea";
            this.chartAvLoss.ChartAreas.Add(chartArea2);
            legend2.Alignment = System.Drawing.StringAlignment.Center;
            legend2.Docking = System.Windows.Forms.DataVisualization.Charting.Docking.Top;
            legend2.Enabled = false;
            legend2.LegendStyle = System.Windows.Forms.DataVisualization.Charting.LegendStyle.Row;
            legend2.Name = "legend";
            this.chartAvLoss.Legends.Add(legend2);
            resources.ApplyResources(this.chartAvLoss, "chartAvLoss");
            this.chartAvLoss.Name = "chartAvLoss";
            series2.BorderWidth = 2;
            series2.ChartArea = "chartArea";
            series2.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series2.Color = System.Drawing.Color.FromArgb(((int)(((byte)(57)))), ((int)(((byte)(106)))), ((int)(((byte)(177)))));
            series2.Legend = "legend";
            series2.LegendText = "Average Loss";
            series2.Name = "seriesLoss";
            this.chartAvLoss.Series.Add(series2);
            this.chartAvLoss.Series.SuspendUpdates();
            // 
            // TrainingForm
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.chartAvLoss);
            this.Controls.Add(this.btnLoad);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.chartAvReward);
            this.Controls.Add(this.textBoxActions);
            this.Controls.Add(this.textBoxInformation);
            this.Controls.Add(this.btnLearning);
            this.Controls.Add(this.btnRunResume);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Name = "TrainingForm";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.OnFormClose);
            ((System.ComponentModel.ISupportInitialize)(this.chartAvReward)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartAvLoss)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnRunResume;
        private System.Windows.Forms.Button btnLearning;
        private System.Windows.Forms.TextBox textBoxInformation;
        private System.Windows.Forms.TextBox textBoxActions;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartAvReward;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.Button btnLoad;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartAvLoss;
    }
}