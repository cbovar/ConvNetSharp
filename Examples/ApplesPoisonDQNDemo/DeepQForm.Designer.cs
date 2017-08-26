namespace DeepQLearning
{
    partial class DeepQForm
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
            this.startLearning = new System.Windows.Forms.Button();
            this.displayBox = new System.Windows.Forms.TextBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.loadNet = new System.Windows.Forms.Button();
            this.saveNet = new System.Windows.Forms.Button();
            this.pauseButton = new System.Windows.Forms.Button();
            this.goSlow = new System.Windows.Forms.Button();
            this.goNormal = new System.Windows.Forms.Button();
            this.goFast = new System.Windows.Forms.Button();
            this.goVeryFast = new System.Windows.Forms.Button();
            this.StopLearning = new System.Windows.Forms.Button();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.canvas = new System.Windows.Forms.Panel();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // startLearning
            // 
            this.startLearning.Location = new System.Drawing.Point(6, 17);
            this.startLearning.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.startLearning.Name = "startLearning";
            this.startLearning.Size = new System.Drawing.Size(100, 22);
            this.startLearning.TabIndex = 0;
            this.startLearning.Text = "Start Learning";
            this.startLearning.UseVisualStyleBackColor = true;
            this.startLearning.Click += new System.EventHandler(this.OnStartLearning);
            // 
            // displayBox
            // 
            this.displayBox.Location = new System.Drawing.Point(6, 248);
            this.displayBox.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.displayBox.Multiline = true;
            this.displayBox.Name = "displayBox";
            this.displayBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.displayBox.Size = new System.Drawing.Size(271, 166);
            this.displayBox.TabIndex = 1;
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.loadNet);
            this.groupBox1.Controls.Add(this.saveNet);
            this.groupBox1.Controls.Add(this.pauseButton);
            this.groupBox1.Controls.Add(this.goSlow);
            this.groupBox1.Controls.Add(this.goNormal);
            this.groupBox1.Controls.Add(this.goFast);
            this.groupBox1.Controls.Add(this.goVeryFast);
            this.groupBox1.Controls.Add(this.StopLearning);
            this.groupBox1.Controls.Add(this.startLearning);
            this.groupBox1.Location = new System.Drawing.Point(732, 434);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox1.Size = new System.Drawing.Size(280, 103);
            this.groupBox1.TabIndex = 3;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Controls";
            // 
            // loadNet
            // 
            this.loadNet.Location = new System.Drawing.Point(141, 72);
            this.loadNet.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.loadNet.Name = "loadNet";
            this.loadNet.Size = new System.Drawing.Size(135, 24);
            this.loadNet.TabIndex = 8;
            this.loadNet.Text = "Load QNetwork";
            this.loadNet.UseVisualStyleBackColor = true;
            this.loadNet.Click += new System.EventHandler(this.OnLoadNet);
            // 
            // saveNet
            // 
            this.saveNet.Location = new System.Drawing.Point(6, 72);
            this.saveNet.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.saveNet.Name = "saveNet";
            this.saveNet.Size = new System.Drawing.Size(130, 24);
            this.saveNet.TabIndex = 7;
            this.saveNet.Text = "Save QNetwork";
            this.saveNet.UseVisualStyleBackColor = true;
            this.saveNet.Click += new System.EventHandler(this.OnSaveNet);
            // 
            // pauseButton
            // 
            this.pauseButton.Location = new System.Drawing.Point(111, 17);
            this.pauseButton.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.pauseButton.Name = "pauseButton";
            this.pauseButton.Size = new System.Drawing.Size(61, 22);
            this.pauseButton.TabIndex = 6;
            this.pauseButton.Text = "Pause";
            this.pauseButton.UseVisualStyleBackColor = true;
            this.pauseButton.Click += new System.EventHandler(this.OnPause);
            // 
            // goSlow
            // 
            this.goSlow.Location = new System.Drawing.Point(212, 44);
            this.goSlow.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.goSlow.Name = "goSlow";
            this.goSlow.Size = new System.Drawing.Size(64, 22);
            this.goSlow.TabIndex = 5;
            this.goSlow.Text = "Go slow";
            this.goSlow.UseVisualStyleBackColor = true;
            this.goSlow.Click += new System.EventHandler(this.OnSlowSpeed);
            // 
            // goNormal
            // 
            this.goNormal.Location = new System.Drawing.Point(141, 44);
            this.goNormal.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.goNormal.Name = "goNormal";
            this.goNormal.Size = new System.Drawing.Size(66, 22);
            this.goNormal.TabIndex = 4;
            this.goNormal.Text = "Go normal";
            this.goNormal.UseVisualStyleBackColor = true;
            this.goNormal.Click += new System.EventHandler(this.OnNormalSpeed);
            // 
            // goFast
            // 
            this.goFast.Location = new System.Drawing.Point(84, 44);
            this.goFast.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.goFast.Name = "goFast";
            this.goFast.Size = new System.Drawing.Size(52, 22);
            this.goFast.TabIndex = 3;
            this.goFast.Text = "Go fast";
            this.goFast.UseVisualStyleBackColor = true;
            this.goFast.Click += new System.EventHandler(this.OnFastSpeed);
            // 
            // goVeryFast
            // 
            this.goVeryFast.Location = new System.Drawing.Point(6, 44);
            this.goVeryFast.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.goVeryFast.Name = "goVeryFast";
            this.goVeryFast.Size = new System.Drawing.Size(74, 22);
            this.goVeryFast.TabIndex = 2;
            this.goVeryFast.Text = "Go very fast";
            this.goVeryFast.UseVisualStyleBackColor = true;
            this.goVeryFast.Click += new System.EventHandler(this.OnVeryFastSpeed);
            // 
            // StopLearning
            // 
            this.StopLearning.Location = new System.Drawing.Point(176, 17);
            this.StopLearning.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.StopLearning.Name = "StopLearning";
            this.StopLearning.Size = new System.Drawing.Size(100, 22);
            this.StopLearning.TabIndex = 1;
            this.StopLearning.Text = "Stop Learning";
            this.StopLearning.UseVisualStyleBackColor = true;
            this.StopLearning.Click += new System.EventHandler(this.OnStopLearning);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.canvas);
            this.groupBox2.Location = new System.Drawing.Point(9, 10);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox2.Size = new System.Drawing.Size(718, 527);
            this.groupBox2.TabIndex = 5;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Visualization";
            // 
            // canvas
            // 
            this.canvas.BackColor = System.Drawing.SystemColors.Info;
            this.canvas.Location = new System.Drawing.Point(4, 17);
            this.canvas.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.canvas.Name = "canvas";
            this.canvas.Size = new System.Drawing.Size(710, 502);
            this.canvas.TabIndex = 0;
            this.canvas.Paint += new System.Windows.Forms.PaintEventHandler(this.PaintCanvas);
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.displayBox);
            this.groupBox3.Location = new System.Drawing.Point(732, 10);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.groupBox3.Size = new System.Drawing.Size(280, 419);
            this.groupBox3.TabIndex = 0;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Output";
            // 
            // DeepQForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1022, 547);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Margin = new System.Windows.Forms.Padding(2, 2, 2, 2);
            this.MaximizeBox = false;
            this.Name = "DeepQForm";
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Show;
            this.Text = "Deep Q Learning Demo";
            this.FormClosed += new System.Windows.Forms.FormClosedEventHandler(this.OnFormClose);
            this.groupBox1.ResumeLayout(false);
            this.groupBox2.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button startLearning;
        private System.Windows.Forms.TextBox displayBox;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Button StopLearning;
        private System.Windows.Forms.Button goSlow;
        private System.Windows.Forms.Button goNormal;
        private System.Windows.Forms.Button goFast;
        private System.Windows.Forms.Button goVeryFast;
        private System.Windows.Forms.Panel canvas;
        private System.Windows.Forms.Button loadNet;
        private System.Windows.Forms.Button saveNet;
        private System.Windows.Forms.Button pauseButton;
    }
}

